"""Publish development assets; research exports require --record explicitly."""
import argparse
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--record', action='store_true', help='Publish record/export.json as a dated snapshot')
parser.add_argument('--source-commit', default='', help='Upstream commit used for this snapshot')
args = parser.parse_args()
frontend = Path(__file__).resolve().parents[1]
root = frontend.parent
asset_dir = frontend / 'assets' / 'scenes'
asset_dir.mkdir(parents=True, exist_ok=True)
(frontend / 'data').mkdir(exist_ok=True)
scenes = []
for source in sorted((root / 'scenes' / 'dev').glob('*.json')):
    scene = json.loads(source.read_text())
    if scene.get('split') != 'dev':
        continue
    views = {'main': scene['image'], **scene.get('image_views', {})}
    if scene.get('image_multiview'):
        views['multiview'] = scene['image_multiview']
    scene['public_views'] = {}
    for label, path in views.items():
        original = (root / path).resolve()
        original.relative_to((root / 'scenes' / 'dev').resolve())
        suffix = '' if label == 'main' else '_' + label.replace(' ', '_')
        name = scene['id'] + suffix + '.png'
        shutil.copyfile(original, asset_dir / name)
        scene['public_views'][label] = 'assets/scenes/' + name
    scenes.append(scene)
if not scenes:
    raise SystemExit('No development scenes found')
(frontend / 'data' / 'scenes.json').write_text(json.dumps(scenes))

demos = [
    ('grasp-baseline', 'Grasp baseline', 'v3_grasp_elliptic_baseline', 'T0-dev-01', 0, 'Single-block grasp smoke test.'),
    ('grasp-lift', 'Vertical lift', 'v3_grasp_lift', 'T1-dev-01', 1, 'A two-block stacking probe with a revised lift stage.'),
    ('grasp-release', 'Release variant', 'v3_grasp_release', 'T1-dev-01', 1, 'A two-block stacking probe with a revised release stage.'),
]
demo_dir = frontend / 'assets' / 'demos'
demo_dir.mkdir(exist_ok=True)
manifest = []
for key, title, method, scene, seed, caption in demos:
    original = root / 'smoke_videos' / method / f'{scene}_s{seed}.mp4'
    if not original.exists():
        continue
    shutil.copyfile(original, demo_dir / f'{key}.mp4')
    manifest.append({'id': key, 'title': title, 'method': method, 'scene': scene, 'seed': seed,
                     'src': f'assets/demos/{key}.mp4', 'caption': caption,
                     'source_path': str(original.relative_to(root))})
(frontend / 'data' / 'demos.json').write_text(json.dumps(manifest))

if args.record:
    record = json.loads((root / 'record' / 'export.json').read_text())
    public = []
    for obj in record:
        item = dict(obj)
        item['data'] = dict(obj['data'])
        item['data'].pop('history', None)
        if item['data'].get('run_fingerprint'):
            fingerprint = item['data']['run_fingerprint']
            item['data']['run_fingerprint'] = {k: fingerprint[k] for k in ['id', 'git_head', 'workspace_sha256', 'captured_at'] if k in fingerprint}
        public.append(item)
    (frontend / 'data' / 'record.json').write_text(json.dumps(public))
    (frontend / 'data' / 'record-meta.json').write_text(json.dumps({'source_commit': args.source_commit,
        'source_repository': 'JonathanLehner/Omnigent-Robotics-Lab', 'source_branch': 'lab-v1',
        'record_updated': max(o.get('updated', o.get('created', 0)) for o in public), 'objects': len(public)}))
print(f'Copied {len(scenes)} development scenes and {len(manifest)} demos. Held-out scenes excluded from frontend.')
