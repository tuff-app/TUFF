#!/usr/bin/env python3
"""Model-free checks against an assembled TUFF app; never launch its GUI."""
import argparse
import fcntl
import plistlib
import json
import os
from pathlib import Path
import re
import subprocess
import socket
import time
import urllib.error
import urllib.request
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def check(app):
    info = plistlib.loads((app / 'Contents/Info.plist').read_bytes())
    version = re.search(r'static let current\s*=\s*"([^"]+)"',
                        (ROOT / 'Sources/TUFFModelCatalog/TUFFVersion.swift').read_text())[1]
    assert info['CFBundleVersion'] == info['CFBundleShortVersionString'] == version
    assert info['SURequireSignedFeed'] is True and info['SUVerifyUpdateBeforeExtraction'] is True
    subprocess.run(['codesign', '--verify', '--deep', '--strict', str(app)], check=True)
    cli = app / 'Contents/Resources/bin/tuff'
    assert subprocess.check_output([str(cli), '--version'], text=True).strip() == f'tuff {version}'
    assert 'tuff prompt' in subprocess.check_output([str(cli), '--help'], text=True)
    # Lightweight host roles share one executable. Family helpers are installed
    # on demand and must never be embedded in the app.
    main = app / 'Contents/MacOS/TUFF'
    for link in ['Resources/bin/TUFFCLI', 'Resources/bin/TUFFServer']:
        path = app / 'Contents' / link
        assert path.is_symlink() and path.resolve() == main.resolve(), link
    for name in ['TUFFCLI', 'TUFFServer']:
        usage = subprocess.check_output([str(app / 'Contents/Resources/bin' / name), '--help'], text=True)
        assert f'usage: {name}' in usage, name
    for name in ['tuff', 'TUFFRepack']:
        assert not (app / 'Contents/Resources/bin' / name).is_symlink(), name
    agent = app / 'Contents/Library/LaunchAgents' / (info['CFBundleIdentifier'] + '.server.plist')
    job = plistlib.loads(agent.read_bytes())
    assert job['BundleProgram'] == 'Contents/Resources/bin/TUFFServer'
    assert job['ProgramArguments'] == ['TUFFServer', '--background']
    assert job['RunAtLoad'] and job['KeepAlive'] == {'SuccessfulExit': False}
    assert not (app / 'Contents/MacOS/TUFFDecodeService').exists()
    assert not (app / 'Contents/MacOS/TUFFInferenceEngine').exists()
    assert not (app / 'Contents/Resources/TUFF_TUFFEngine.bundle').exists()
    assert (app / 'Contents/Resources/engine-catalog.json').is_file()
    for name in ['TUFF_TUFFAppCore.bundle', 'SwiftMath_SwiftMath.bundle']:
        assert (app / 'Contents/Resources' / name).is_dir(), name
    with tempfile.TemporaryDirectory(prefix='tuff-package-mismatch-') as tmp:
        result = subprocess.run([str(ROOT / 'Scripts/package_app.sh'), '0.0.0', tmp], capture_output=True, text=True)
        assert result.returncode == 64 and 'disagrees' in result.stderr
        assert not list(Path(tmp).iterdir()), 'version mismatch must fail before writing artifacts'
    with tempfile.TemporaryDirectory(prefix='tuff-package-idle-') as tmp:
        env = os.environ.copy()
        env.update(TUFF_LOCAL_BUILD='1', TUFF_APPLICATION_SUPPORT_ROOT=tmp,
                   TUFF_SERVING_BUNDLE_ID=info['CFBundleIdentifier'])
        host = app / 'Contents/Resources/bin/TUFFServer'
        idle = subprocess.run([str(host), '--private-host', '--serving-bundle-id',
                               info['CFBundleIdentifier']], env=env, capture_output=True, text=True, timeout=90)
        assert idle.returncode == 0, f'idle private host exited {idle.returncode}: {idle.stderr}'
        print('Packaged private host retired idle with exit 0')
    print(f'Packaged TUFF {version}: signature, version, CLI, roles, agent, signed-feed policy and resources passed')

def check_standalone_serve(app, support):
    """Opt-in real-model regression: a fresh profile must launch its own host."""
    env = os.environ.copy()
    bundle = plistlib.loads((app / 'Contents/Info.plist').read_bytes())['CFBundleIdentifier']
    env.update(TUFF_LOCAL_BUILD='1', TUFF_APPLICATION_SUPPORT_ROOT=str(support),
               TUFF_SERVING_BUNDLE_ID=bundle,
               TUFF_ENGINE_CATALOG_PATH=str(app / 'Contents/Resources/engine-catalog.json'))
    lock = support / 'TUFF/Server/serving-host.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600), 'a') as lease:
        # Do not pass by attaching to a previously started host.
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lease, fcntl.LOCK_UN)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [str(app / 'Contents/Resources/bin/tuff'), 'serve', '--port', str(port),
               '--models-root', str(support / 'TUFF/Models'), '--default-model', 'gemma-4-e2b-it']
    with tempfile.TemporaryFile(mode='w+') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            url = f'http://127.0.0.1:{port}'
            deadline = time.monotonic() + 90
            while True:
                try:
                    with urllib.request.urlopen(url + '/v1/models', timeout=2):
                        break
                except (OSError, urllib.error.URLError):
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError('packaged server did not start')
                    time.sleep(0.2)
            body = json.dumps({'model': 'gemma-4-e2b-it', 'max_tokens': 16, 'temperature': 0,
                               'messages': [{'role': 'user', 'content': 'What is 2 plus 2? Answer with one number.'}]}).encode()
            request = urllib.request.Request(url + '/v1/chat/completions', data=body,
                                             headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=600) as response:
                answer = json.load(response)['choices'][0]['message']['content']
                assert response.status == 200 and '4' in answer, answer
            print('Standalone packaged tuff serve launched a fresh host and answered: ' + answer)
        finally:
            process.terminate()
            process.wait(timeout=30)
            log.seek(0)
            print(log.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', required=True, type=Path)
    parser.add_argument('--real-model-support', type=Path,
                        help='isolated support root containing verified Gemma E2B')
    args = parser.parse_args()
    app = args.app.resolve()
    check(app)
    if args.real_model_support:
        check_standalone_serve(app, args.real_model_support.resolve())
