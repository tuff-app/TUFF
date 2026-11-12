#!/usr/bin/env python3
"""Capture the same isolated render cases on a real macOS 26 or 27 host."""
import argparse, hashlib, itertools, json, os, pathlib, platform, subprocess
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('output', type=pathlib.Path)
p.add_argument('--smoke', action='store_true', help='one case to validate the harness')
a = p.parse_args()
assert platform.system() == 'Darwin', 'Run on the actual macOS version under test'
a.output = a.output.resolve()
a.output.mkdir(parents=True, exist_ok=True)
(a.output / 'run.json').unlink(missing_ok=True)
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
subprocess.run(['bash', 'Scripts/test.sh', '--build-only'], check=True)
cases = [('light', 'chat', 'expanded', 0, 1080)] if a.smoke else itertools.product(
    ['light', 'dark'], ['chat', 'settings'], ['expanded', 'hidden'], [0, .7, 1], [1080, 640])
for theme, screen, sidebar, opacity, width in cases:
    key = f'{theme}-{screen}-{sidebar}-{opacity:g}-{width}'
    target = a.output / key
    env = dict(os.environ, TUFF_VISUAL_OUTPUT=str(target), TUFF_VISUAL_APPEARANCE=theme,
        TUFF_VISUAL_WIDTH=str(width), TUFF_VISUAL_TRANSPARENCY=str(opacity),
        TUFF_OPEN_SCREEN=screen, TUFF_DEBUG_SIDEBAR=sidebar,
        TUFF_APPLICATION_SUPPORT_ROOT=str(a.output / 'isolated-support'))
    subprocess.run(['bash', 'Scripts/test.sh', '--skip-build', '--filter', 'CrossOSVisualTests'], env=env, check=True)
    assert (target / 'render.png').is_file(), f'Capture missing: {key}'
(a.output / 'run.json').write_text(json.dumps({'commit': revision, 'os': platform.mac_ver()[0],
    'xcode': subprocess.check_output(['xcodebuild', '-version'], text=True),
    'dirty': bool(subprocess.check_output(['git', 'diff', 'HEAD', '--', 'Sources'], text=True)),
    'smoke': a.smoke, 'fixtureSHA256': hashlib.sha256(pathlib.Path('Tests/TUFFApp/MacPresentation/CrossOSVisualTests.swift').read_bytes()).hexdigest()}, indent=2) + '\n')
