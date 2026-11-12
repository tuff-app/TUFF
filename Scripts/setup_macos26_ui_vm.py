#!/usr/bin/env python3
"""Provision a separate macOS 26 UI sandbox on a storage volume you choose."""
import argparse, json, os, pathlib, shutil, subprocess
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--store', required=True, type=pathlib.Path, help='VM/cache directory; external SSD recommended')
p.add_argument('--app', required=True, type=pathlib.Path, help='Packaged TUFF.app to test unchanged')
p.add_argument('--create', action='store_true', help='Download VM and open it using an installed Tart')
a = p.parse_args()
a.store = a.store.expanduser().resolve()
a.app = a.app.expanduser().resolve()
assert (a.app / 'Contents/Info.plist').is_file(), 'Supply a packaged TUFF.app'
a.store.mkdir(parents=True, exist_ok=True)
free = shutil.disk_usage(a.store).free / 2**30
# Conservative project budget, not an assertion about the image's final size.
assert free >= 80, f'{free:.1f} GiB free; reserve at least 80 GiB for download, cache and guest growth'
assert not a.store.is_relative_to(a.app), 'Keep VM storage outside the app'
print(f'Preflight passed: {free:.1f} GiB free at {a.store}')
if not a.create:
    print('Add --create to download and boot the VM. Nothing installed or downloaded.')
    raise SystemExit(0)
tart = shutil.which('tart') or str(pathlib.Path.home() / 'Downloads/TUFF UI Testing/tools/tart.app/Contents/MacOS/tart')
assert pathlib.Path(tart).is_file(), 'Install Tart from its official release archive first'
env = dict(os.environ, TART_HOME=str(a.store / 'tart'))
name = 'tuff-ui-macos26'
image = 'ghcr.io/cirruslabs/macos-tahoe-base:latest'
subprocess.run([tart, 'clone', image, name], env=env, check=True)
shared = a.store / 'shared-app'
shared.mkdir(exist_ok=True)
assert not (shared / 'TUFF.app').exists(), 'An app is already staged here; preserve it and choose a fresh store'
subprocess.run(['ditto', str(a.app), str(shared / 'TUFF.app')], check=True)
(a.store / 'setup.json').write_text(json.dumps({'image': image, 'vm': name, 'app': str(a.app)}, indent=2) + '\n')
# Expose only the packaged app, never the checkout, credentials or real models.
subprocess.run([tart, 'run', '--dir=tuff-app:' + str(shared) + ':ro', name], env=env, check=True)
