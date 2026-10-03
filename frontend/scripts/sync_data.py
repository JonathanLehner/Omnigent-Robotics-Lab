"""Copy public development benchmarks and an optional explicit research export."""
import json
import shutil
from pathlib import Path

frontend = Path(__file__).resolve().parents[1]
root = frontend.parent
asset_dir = frontend / 'assets' / 'scenes'
asset_dir.mkdir(parents=True, exist_ok=True)
(frontend / 'data').mkdir(exist_ok=True)
scenes = []
for source in sorted((root / 'scenes' / 'dev').glob('*.json')):
    scene = json.loads(source.read_text())
    scenes.append(scene)
    shutil.copyfile(source.with_suffix('.png'), asset_dir / (scene['id'] + '.png'))
if not scenes:
    raise SystemExit('No development scenes found in ../scenes/dev')
(frontend / 'data' / 'scenes.json').write_text(json.dumps(scenes))
print(f'Copied {len(scenes)} development benchmarks. Held-out scenes are excluded.')
