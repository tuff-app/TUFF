#!/usr/bin/env python3
"""Check that a new TUFF still works on a profile an older TUFF left behind.

TUFF 10.0.0 passed every test on fresh profiles and still failed every model
load for people updating from 9.x, because 9.x had left its Server folder
readable by others. This runs the older release's `tuff` against a scoped
Application Support root, adds the files the older app's GUI writes, then
runs the new build's `tuff prompt`, `tuff serve` and Background API against
the same root while the older `tuff serve` is still running. From 10.0 on, an
updated app has to take over the serving host the old one left running.

The model is an APFS clone of an installed one, so it takes no extra space.
Nothing outside the work directory is touched, and it's deleted at the end
unless --keep is given. Engines are downloaded the same way the apps would
download them; pass TUFF_ENGINE_CATALOG_PATH in the environment to test
unreleased engines.

    Scripts/validate_upgrade.py --from /path/to/9.6.0/TUFF.app \\
        --to dist/TUFF.app
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

DEFAULT_MODELS = Path.home() / 'Library/Application Support/TUFF/Models'
PROMPT = 'Say hi in three words.'

# What the 9.x app's GUI writes into TUFF/Server. It can't be run against a
# scoped root without touching the real defaults and login items, so these
# are copied from a real profile that was updated from 9.x: the folder and
# both files were left readable by others.
GUI_SERVER_FILES = {
    'background-server.json': json.dumps({
        'defaultModel': 'gemma-4-e2b-it', 'enabled': True, 'port': 8080,
        'queueLimit': 4, 'unloadDelay': 300, 'version': 1}, indent=2) + '\n',
    'registered-build': '9.6.0\n',
}


class Failure(Exception):
    pass


def tool(app):
    path = Path(app) / 'Contents/Resources/bin/tuff'
    if not path.is_file():
        raise Failure(f'no bundled tuff in {app}')
    return path


def version(app):
    info = Path(app) / 'Contents/Info.plist'
    result = subprocess.run(['plutil', '-extract', 'CFBundleShortVersionString', 'raw', str(info)],
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def clone_model(source, models):
    target = models / source.name
    models.mkdir(parents=True, exist_ok=True)
    subprocess.run(['cp', '-cR', str(source), str(target)], check=True)
    # The install receipt is bound to the directory it was verified in.
    receipt = target / 'verified-install.json'
    data = json.loads(receipt.read_text())
    data['modelDirectoryPath'] = str(target.resolve())
    receipt.write_text(json.dumps(data, indent=2) + '\n')


def environment(root, home):
    env = os.environ.copy()
    env.update(TUFF_APPLICATION_SUPPORT_ROOT=str(root), HOME=str(home),
               CFFIXED_USER_HOME=str(home))
    return env


def prompt(app, model, env, cwd, log):
    result = subprocess.run(
        [str(tool(app)), 'prompt', PROMPT, '--model', model, '--max-new', '24',
         '--temperature', '0'], stdin=subprocess.DEVNULL, capture_output=True,
        text=True, env=env, cwd=cwd, timeout=900)
    log.write_text(result.stdout + result.stderr)
    answer = [line for line in result.stdout.splitlines() if line.strip() and not line.startswith('[')]
    if result.returncode != 0 or not answer:
        raise Failure(f'`tuff prompt` exited {result.returncode}; see {log}')
    return ' '.join(answer)


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def request(port, path, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=data,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b'{}')
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b'{}')


def serve(app, model, env, cwd, log):
    port = free_port()
    command = [str(tool(app)), 'serve', '--port', str(port), '--default-model', model,
               '--unload-after', '600']
    return answer_from_server(command, port, 'default', env, cwd, log, '`tuff serve`')


def background(app, root, env, cwd):
    # The Background API reads its port and default model from the
    # background-server.json the older app left, and logs inside the scoped root.
    settings_path = root / 'TUFF/Server/background-server.json'
    settings = json.loads(settings_path.read_text())
    port = free_port()
    settings['port'] = port
    settings_path.write_text(json.dumps(settings, indent=2) + '\n')
    settings_path.chmod(0o644)
    command = [str(Path(app) / 'Contents/Resources/bin/TUFFServer'), '--background']
    log = root / 'TUFF/Logs/background-server.log'
    return answer_from_server(command, port, 'default', env, cwd, log, 'Background API',
                              capture=False)


def answer_from_server(command, port, model, env, cwd, log, name, capture=True):
    process = start_server(command, port, env, cwd, log, name, capture)
    try:
        return ask(port, model, log)
    finally:
        stop(process)


def start_server(command, port, env, cwd, log, name, capture=True):
    log.parent.mkdir(parents=True, exist_ok=True)
    with (log.open('wb') if capture else open(os.devnull, 'wb')) as output:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                   stderr=subprocess.STDOUT, env=env, cwd=cwd)
    for _ in range(240):
        if process.poll() is not None:
            raise Failure(f'{name} exited {process.returncode}; see {log}')
        try:
            if request(port, '/health', timeout=2)[0] == 200:
                return process
        except OSError:
            pass
        time.sleep(0.5)
    stop(process)
    raise Failure(f'{name} never became healthy; see {log}')


def ask(port, model, log):
    status, body = request(port, '/v1/chat/completions', {
        'model': model, 'messages': [{'role': 'user', 'content': PROMPT}],
        'max_tokens': 24, 'temperature': 0})
    if status != 200:
        message = body.get('error', {}).get('message', body)
        raise Failure(f'chat completion returned {status}: {message}; see {log}')
    content = body['choices'][0]['message'].get('content') or ''
    if not content.strip():
        raise Failure(f'chat completion was empty; see {log}')
    return content.strip()


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


class OldServer:
    """The older `tuff serve`, left running the way an old Background API or a
    terminal keeps running through an update."""

    def __init__(self, app, model, env, cwd, log):
        self.process = None
        self.args = (app, model, env, cwd, log)

    def start(self):
        app, model, env, cwd, log = self.args
        port = free_port()
        self.process = start_server(
            [str(tool(app)), 'serve', '--port', str(port), '--default-model', model,
             '--unload-after', '600'], port, env, cwd, log, 'older `tuff serve`')
        return ask(port, 'default', log)

    def stop(self):
        if self.process:
            stop(self.process)


def add_gui_files(root):
    server = root / 'TUFF/Server'
    server.mkdir(parents=True, exist_ok=True)
    for name, contents in GUI_SERVER_FILES.items():
        path = server / name
        if not path.exists():
            path.write_text(contents)
        path.chmod(0o644)
    server.chmod(0o755)


def check_private(path):
    mode = stat.S_IMODE(path.lstat().st_mode)
    if mode & 0o077:
        raise Failure(f'{path} is still mode {mode:o} after the new build ran')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--from', dest='old', required=True, help='older TUFF.app (for example 9.6.0)')
    parser.add_argument('--to', dest='new', required=True, help='TUFF.app being released')
    parser.add_argument('--model', default='gemma4-e2b', help='installed model folder name, without .gturbo')
    parser.add_argument('--models-root', type=Path, default=DEFAULT_MODELS)
    parser.add_argument('--work', type=Path, help='work directory (default: a new temporary one)')
    parser.add_argument('--keep', action='store_true', help='keep the work directory')
    args = parser.parse_args()
    args.old, args.new = Path(args.old).resolve(), Path(args.new).resolve()

    source = args.models_root / f'{args.model}.gturbo'
    if not source.is_dir():
        sys.exit(f'{source} is not installed')
    work = args.work or Path(tempfile.mkdtemp(prefix='tuff-upgrade-'))
    work.mkdir(parents=True, exist_ok=True)
    work = work.resolve()
    root, home, cwd = work / 'support', work / 'home', work / 'cwd'
    for path in (root, home, cwd):
        path.mkdir(exist_ok=True)
    env = environment(root, home)
    steps = []
    ok = True
    old_server = None
    try:
        old, new = version(args.old), version(args.new)
        print(f'upgrade {old} -> {new}, model {args.model}, work {work}')
        clone_model(source, root / 'TUFF/Models')
        old_server = OldServer(args.old, args.model, env, cwd, work / 'old-serve.log')
        for label, action in [
            (f'{old} tuff prompt', lambda: prompt(args.old, args.model, env, cwd, work / 'old-prompt.log')),
            (f'{old} app files', lambda: add_gui_files(root)),
            (f'{old} tuff serve, left running', old_server.start),
            (f'{new} tuff prompt', lambda: prompt(args.new, args.model, env, cwd, work / 'new-prompt.log')),
            (f'{new} tuff serve', lambda: serve(args.new, args.model, env, cwd, work / 'new-serve.log')),
            (f'{new} Background API', lambda: background(args.new, root, env, cwd)),
            (f'{new} left Server private', lambda: check_private(root / 'TUFF/Server')),
        ]:
            try:
                result = action()
                steps.append((label, True, result))
                print(f'  pass  {label}' + (f': {result!r}' if isinstance(result, str) else ''))
            except Failure as error:
                steps.append((label, False, str(error)))
                print(f'  FAIL  {label}: {error}')
                ok = False
                break
    finally:
        if old_server:
            old_server.stop()
        if args.keep or not ok:
            print(f'kept {work}')
        else:
            shutil.rmtree(work)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
