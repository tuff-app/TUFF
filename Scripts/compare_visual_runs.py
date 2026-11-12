#!/usr/bin/env python3
"""Make an offline side-by-side/overlay review from two visual_matrix runs."""
import argparse, base64, html, json, pathlib
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('left', type=pathlib.Path)
p.add_argument('right', type=pathlib.Path)
p.add_argument('output', type=pathlib.Path)
a = p.parse_args()
l, r = [json.loads((root / 'run.json').read_text()) for root in [a.left, a.right]]
assert l['commit'] == r['commit'], 'Compare the same source commit'
assert not l['dirty'] and not r['dirty'], 'Source modifications invalidate the comparison'
left = {p.parent.name: p for p in a.left.glob('*/render.png')}
right = {p.parent.name: p for p in a.right.glob('*/render.png')}
assert left and left.keys() == right.keys(), 'Both runs must contain exactly the same non-empty case matrix'
assert l['fixtureSHA256'] == r['fixtureSHA256'], 'Capture fixtures differ'
assert l['xcode'] == r['xcode'], 'Use the same Xcode/SDK on both hosts'
assert l['smoke'] == r['smoke'], 'Cannot compare smoke to full matrix'
parts = ['<!doctype html><meta charset="utf-8"><title>TUFF cross-macOS review</title><style>body{font:16px system-ui;margin:30px;background:#eee;color:#222}section{margin:40px 0} .pair{display:flex;gap:12px}.pair img{width:49%;object-fit:contain;align-self:start}.overlay{position:relative;width:75%}.overlay img{width:100%;display:block}.overlay img+img{position:absolute;inset:0;opacity:.5}summary{cursor:pointer}</style>',
    '<h1>TUFF cross-macOS render review</h1><p>' + html.escape(str(l) + ' compared with ' + str(r)) + '</p>',
    '<p>Inspect differences manually. This is not a pixel-equality pass/fail gate. Offscreen content excludes native window corners, traffic lights and desktop glass.</p>']
for key in sorted(left):
    lm, rm = [json.loads((path.parent / 'metadata.json').read_text()) for path in [left[key], right[key]]]
    assert (lm['pixelWidth'], lm['pixelHeight']) == (rm['pixelWidth'], rm['pixelHeight']), f'Scale mismatch: {key}'
    urls = ['data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() for path in [left[key], right[key]]]
    parts.append('<section><h2>' + html.escape(key) + '</h2><div class="pair">' + ''.join('<img alt="' + label + '" src="' + url + '">' for label, url in zip(['Left', 'Right'], urls)) + '</div><details><summary>Overlay comparison</summary><label>Right opacity <input aria-label="Right image opacity" type="range" min="0" max="1" step=".1" value=".5" oninput="this.parentElement.nextElementSibling.lastElementChild.style.opacity=this.value"></label><div class="overlay">' + ''.join('<img alt="Overlay" src="' + url + '">' for url in urls) + '</div></details></section>')
a.output.write_text('\n'.join(parts))
print(a.output.resolve())
