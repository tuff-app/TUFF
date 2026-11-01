#!/usr/bin/env python3
"""Soak the shared serving host with mixed traffic from two clients at once.

From 10.0, `tuff serve`, the Background API and the app share one serving
host. This starts `tuff serve` and the Background API against a scratch
profile, so both attach to the same host, then runs several workers for a
set time. Each request picks a model, a client, streaming or not, and
sometimes a follow-up turn or a stream dropped part way.

Every finished greedy answer must equal the answer the same request got on
its own before the load started, so sessions can't bleed into each other.
Busy answers (`queue_full`, `model_busy`, `model_memory_busy`) are counted,
not failed. Anything else that isn't a 200, a request that hangs, a server
that exits, or a TUFF process left running afterwards fails the run.

Models are APFS clones of installed ones, so they take no extra space.
Nothing outside the work directory is touched; it's deleted at the end
unless --keep is given or the run fails.

    Scripts/validate_serving_soak.py --app /path/to/TUFF.app --minutes 10
"""
import argparse
import http.client
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_upgrade import (DEFAULT_MODELS, Failure, clone_model, environment,  # noqa: E402
                              free_port, start_server, stop, tool)

BUSY = {'queue_full', 'model_busy', 'model_memory_busy'}
PROMPTS = [
    'What is the capital of Japan? Answer in one sentence.',
    'Name three primary colours.',
    'Why is the sea salty? Two sentences.',
    'Write a haiku about a lighthouse.',
]
FOLLOW_UP = 'Now say that again in five words or fewer.'


def post(port, body, timeout):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    try:
        connection.request('POST', '/v1/chat/completions', json.dumps(body),
                           {'Content-Type': 'application/json'})
        response = connection.getresponse()
        data = response.read()
        return response.status, json.loads(data or b'{}')
    finally:
        connection.close()


def stream(port, body, timeout, drop_after=None):
    """Returns (status, text, finished, error code). With drop_after, closes
    the connection after that many content chunks."""
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    try:
        connection.request('POST', '/v1/chat/completions', json.dumps(dict(body, stream=True)),
                           {'Content-Type': 'application/json'})
        response = connection.getresponse()
        if response.status != 200:
            payload = json.loads(response.read() or b'{}')
            return response.status, '', False, payload.get('error', {}).get('code')
        text, chunks, finished = '', 0, False
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                break
            event = json.loads(data)
            if 'error' in event:
                return 200, text, False, event['error'].get('code', 'stream_error')
            for choice in event.get('choices', []):
                delta = choice.get('delta', {}).get('content') or ''
                if delta:
                    text += delta
                    chunks += 1
                if choice.get('finish_reason'):
                    finished = True
            if drop_after is not None and chunks >= drop_after:
                return 200, text, False, 'dropped'
        return 200, text, finished, None
    finally:
        connection.close()


def body(model, messages):
    return {'model': model, 'messages': messages, 'max_tokens': 48, 'temperature': 0}


class Ledger:
    def __init__(self):
        self.lock = threading.Lock()
        self.counts = {}
        self.failures = []

    def count(self, key):
        with self.lock:
            self.counts[key] = self.counts.get(key, 0) + 1

    def fail(self, message):
        with self.lock:
            self.failures.append(message)


def baseline(ports, models, timeout):
    answers = {}
    for model in models:
        for prompt in PROMPTS:
            status, response = post(ports[0], body(model, [{'role': 'user', 'content': prompt}]), timeout)
            if status != 200:
                raise Failure(f'baseline {model} returned {status}: {response}')
            first = response['choices'][0]['message']['content']
            status, response = post(ports[0], body(model, [
                {'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': first},
                {'role': 'user', 'content': FOLLOW_UP}]), timeout)
            if status != 200:
                raise Failure(f'baseline follow-up {model} returned {status}: {response}')
            answers[(model, prompt)] = (first, response['choices'][0]['message']['content'])
    return answers


def worker(index, ports, models, answers, deadline, timeout, ledger, seed):
    rng = random.Random(seed + index)
    while time.monotonic() < deadline:
        model, prompt = rng.choice(models), rng.choice(PROMPTS)
        port = rng.choice(ports)
        first, second = answers[(model, prompt)]
        messages = [{'role': 'user', 'content': prompt}]
        kind = rng.choice(['plain', 'stream', 'follow-up', 'drop'])
        label = f'worker {index} {kind} {model} port {port}'
        try:
            if kind == 'drop':
                status, _, _, code = stream(port, body(model, messages), timeout,
                                            drop_after=rng.randint(1, 4))
                ledger.count('dropped' if code == 'dropped' else (code or 'short stream'))
                if code not in BUSY | {'dropped', None}:
                    ledger.fail(f'{label}: {status} {code}')
                continue
            if kind == 'stream':
                status, text, finished, code = stream(port, body(model, messages), timeout)
                expected = first
            else:
                if kind == 'follow-up':
                    messages += [{'role': 'assistant', 'content': first},
                                 {'role': 'user', 'content': FOLLOW_UP}]
                status, response = post(port, body(model, messages), timeout)
                code = response.get('error', {}).get('code') if status != 200 else None
                text = response['choices'][0]['message']['content'] if status == 200 else ''
                finished = status == 200
                expected = second if kind == 'follow-up' else first
            if code in BUSY:
                ledger.count(code)
                continue
            if status != 200 or code or not finished:
                ledger.fail(f'{label}: {status} {code} finished={finished}')
                continue
            if text != expected:
                ledger.fail(f'{label}: answer changed under load\n  expected {expected!r}\n  got      {text!r}')
                continue
            ledger.count('ok')
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as error:
            ledger.fail(f'{label}: {type(error).__name__}: {error}')


def recover(port, model, expected, timeout, within=60):
    """Seconds until the port gives the expected answer again, or None."""
    started = time.monotonic()
    while time.monotonic() - started < within:
        try:
            status, response = post(port, body(model, [{'role': 'user', 'content': PROMPTS[0]}]), timeout)
            if status == 200 and response['choices'][0]['message']['content'] == expected:
                return time.monotonic() - started
        except (OSError, http.client.HTTPException, json.JSONDecodeError):
            pass
        time.sleep(1)
    return None


def tuff_processes(*paths):
    """This run's processes: its engines live under the work directory and
    its servers and host run from the app being soaked."""
    lines = set()
    for path in paths:
        # /tmp is a link to /private/tmp; command lines can show either.
        pattern = str(path).removeprefix('/private')
        result = subprocess.run(['pgrep', '-fl', pattern], capture_output=True, text=True)
        lines.update(line for line in result.stdout.splitlines()
                     if 'pgrep' not in line and line.split()[0] != str(os.getpid()))
    return sorted(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--app', required=True, help='TUFF.app to soak')
    parser.add_argument('--models', default='gemma4-e2b,bonsai-2-2b',
                        help='installed model folder names, without .gturbo')
    parser.add_argument('--models-root', type=Path, default=DEFAULT_MODELS)
    parser.add_argument('--minutes', type=float, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=600, help='seconds before a request counts as hung')
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--keep', action='store_true')
    args = parser.parse_args()
    app = Path(args.app).resolve()
    names = args.models.split(',')

    work = (args.work or Path(tempfile.mkdtemp(prefix='tuff-soak-'))).resolve()
    root, home, cwd = work / 'support', work / 'home', work / 'cwd'
    for path in (root, home, cwd):
        path.mkdir(parents=True, exist_ok=True)
    env = environment(root, home)
    servers = []
    ok = False
    try:
        for name in names:
            source = args.models_root / f'{name}.gturbo'
            if not source.is_dir():
                raise Failure(f'{source} is not installed')
            clone_model(source, root / 'TUFF/Models')
        serve_port, background_port = free_port(), free_port()
        servers.append(start_server(
            [str(tool(app)), 'serve', '--port', str(serve_port), '--unload-after', '600',
             '--queue-limit', '8'], serve_port, env, cwd, work / 'serve.log', '`tuff serve`'))
        connection = http.client.HTTPConnection('127.0.0.1', serve_port, timeout=10)
        connection.request('GET', '/v1/models')
        listed = json.loads(connection.getresponse().read())
        connection.close()
        models = [entry['id'] for entry in listed.get('data', [])
                  if entry['id'] != 'default'][:len(names)]
        if len(models) < len(names):
            raise Failure(f'expected {len(names)} models, server lists {models}')
        settings = root / 'TUFF/Server/background-server.json'
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text(json.dumps({'defaultModel': models[0], 'enabled': True, 'port': background_port,
                                        'queueLimit': 8, 'unloadDelay': 600, 'version': 1}, indent=2))
        servers.append(start_server(
            [str(app / 'Contents/Resources/bin/TUFFServer'), '--background'], background_port,
            env, cwd, root / 'TUFF/Logs/background-server.log', 'Background API', capture=False))
        ports = [serve_port, background_port]
        print(f'soak {app}: models {models}, {args.workers} workers, {args.minutes} min, work {work}')
        answers = baseline(ports, models, args.timeout)
        print(f'  baseline: {len(answers)} prompt and model pairs')

        ledger = Ledger()
        deadline = time.monotonic() + args.minutes * 60
        threads = [threading.Thread(target=worker, args=(i, ports, models, answers, deadline,
                                                        args.timeout, ledger, args.seed))
                   for i in range(args.workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(args.minutes * 60 + args.timeout + 60)
            if thread.is_alive():
                ledger.fail(f'{thread.name} never finished')
        for server, name in zip(servers, ['`tuff serve`', 'Background API']):
            if server.poll() is not None:
                ledger.fail(f'{name} exited with {server.returncode} during the soak')
        print('  results: ' + ', '.join(f'{key} {value}' for key, value in sorted(ledger.counts.items())))

        # Sessions from the soak must not stop the host from answering afterwards.
        status, response = post(serve_port, body(models[0], [{'role': 'user', 'content': PROMPTS[0]}]),
                                args.timeout)
        if status != 200 or response['choices'][0]['message']['content'] != answers[(models[0], PROMPTS[0])][0]:
            ledger.fail(f'after the soak: {status} {response}')

        # A crashed engine, then a crashed host, must not take either client
        # down for good: the next requests relaunch them.
        for target in ('TUFFInferenceEngine', '--private-host'):
            victims = [line.split()[0] for line in tuff_processes(work, app) if target in line]
            if not victims:
                ledger.fail(f'no {target} process to stop')
                continue
            for pid in victims:
                os.kill(int(pid), 9)
            for port, name in zip(ports, ['`tuff serve`', 'Background API']):
                recovered = recover(port, models[0], answers[(models[0], PROMPTS[0])][0], args.timeout)
                if recovered is None:
                    ledger.fail(f'{name} did not recover after {target} was killed')
                else:
                    ledger.count(f'recovered after {target.strip("-")} kill')
                    print(f'  {name} answered again {recovered:.1f} s after {target} was killed')

        for server in servers:
            stop(server)
        servers = []
        # The host and its engines retire after 20 idle seconds.
        for _ in range(60):
            left = tuff_processes(work, app)
            if not left:
                break
            time.sleep(1)
        else:
            ledger.fail('processes still running a minute after the servers stopped:\n  '
                        + '\n  '.join(left))
        for failure in ledger.failures:
            print(f'  FAIL  {failure}')
        ok = not ledger.failures and ledger.counts.get('ok', 0) > 0
        print('  pass' if ok else '  FAIL')
    except Failure as error:
        print(f'  FAIL  {error}')
    finally:
        for server in servers:
            stop(server)
        if args.keep or not ok:
            print(f'kept {work}')
        else:
            shutil.rmtree(work, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
