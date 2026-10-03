"""Benchmark scenes: target structures by tier, start layouts, rendered target pictures, dev/held-out split.

Structure spec (structure frame = build site on the ground, origin at the center of the bottom layer, x away from the robot, z up; pos is the block center):
  {"blocks": [{"id", "type": cube|brick, "color", "pos": [x, y, z], "yaw": deg, "on": [ids it rests on]}]}
"""

import json
from pathlib import Path

import numpy as np

from lab import sim

ROOT = Path(__file__).resolve().parent.parent
SCENES = ROOT / "scenes"
H = sim.BLOCK_TYPES["cube"][2]
CUBE_W, BRICK_W = sim.BLOCK_TYPES["cube"][1], sim.BLOCK_TYPES["brick"][1]
GAP = 0.01


def _b(t, y, level, dx=0.0):
    return {"type": t, "pos": [dx, y, H / 2 + level * H], "yaw": 0.0}


def templates(tier: str, rng, variant: int | None = None) -> tuple[str, list[dict]]:
    """Target structures per tier. Small random offsets keep every scene different."""
    s = CUBE_W + GAP
    o = float(rng.uniform(-0.03, 0.03))  # stack offset within stable bounds
    choices = {
        "T0": {"single": [_b(str(rng.choice(["cube", "brick"])), 0, 0)]},
        "T1": {
            "stack": [_b("cube", 0, 0), _b("cube", o, 1)],
            "cube_on_brick": [_b("brick", 0, 0), _b("cube", 2 * o, 1)],
        },
        "T2": {
            "tower": [_b("cube", 0, 0), _b("cube", o / 2, 1), _b("cube", 0, 2)],
            "wall": [_b("cube", -s, 0), _b("cube", 0, 0), _b("cube", s, 0)],
            "bridge": [_b("cube", -s / 2, 0), _b("cube", s / 2, 0), _b("brick", 0, 1)],
        },
        "T3": {
            "L": [_b("cube", -s / 2, 0), _b("cube", s / 2, 0), _b("cube", -s / 2, 1), _b("cube", -s / 2, 2)],
            "T": [_b("cube", 0, 0), _b("cube", 0, 1), _b("brick", 0, 2)],
            "pyramid": [_b("cube", -s, 0), _b("cube", 0, 0), _b("cube", s, 0), _b("cube", -s / 2, 1), _b("cube", s / 2, 1),
                        _b("cube", 0, 2)],
        },
    }[tier]
    names = sorted(choices)
    name = names[variant % len(names)] if variant is not None else str(rng.choice(names))
    return name, choices[name]


def support_relations(blocks):
    """Fill `on`: A rests on B if A sits one layer above B and their footprints overlap in y."""
    for a in blocks:
        a["on"] = [
            b["id"] for b in blocks
            if abs(a["pos"][2] - b["pos"][2] - H) < 0.01
            and abs(a["pos"][1] - b["pos"][1]) < (sim.BLOCK_TYPES[a["type"]][1] + sim.BLOCK_TYPES[b["type"]][1]) / 2 - 0.005
        ]
    return blocks


def oracle_order(blocks):
    return [b["id"] for b in sorted(blocks, key=lambda b: (round(b["pos"][2], 3), b["pos"][1]))]


def to_world(b, pos=None, yaw=None):
    p = np.asarray(pos if pos is not None else b["pos"], float)
    return {"id": b["id"], "type": b["type"], "color": b["color"], "pos": list(sim.SITE + p),
            "quat": list(sim.quat_mul([1, 0, 0, 0], _yawq(yaw if yaw is not None else b["yaw"])))}


def _yawq(deg):
    h = np.radians(deg) / 2
    return [np.cos(h), 0, 0, np.sin(h)]


def sample_start(blocks, rng, tries=200):
    """Lay blocks out on the ground beside the build site, handles facing the robot."""
    placed = []
    for b in blocks:
        w = sim.BLOCK_TYPES[b["type"]][1]
        for _ in range(tries):
            x, y = rng.uniform(0.40, 0.64), rng.choice([-1, 1]) * rng.uniform(0.2, 0.44)
            if all(abs(x - p["x"]) > 0.15 or abs(y - p["y"]) > (w + p["w"]) / 2 + 0.03 for p in placed):
                placed.append({"id": b["id"], "x": float(x), "y": float(y), "w": w})
                break
        else:
            return None
    return [{"id": p["id"], "pos": [p["x"], p["y"], H / 2], "yaw": 0.0} for p in placed]


def reachable(blocks, start) -> bool:
    """Every grasp (start handle) and place (target handle) must have an IK solution at some grasp pitch."""
    w = sim.World([])
    for b in blocks:
        s = next(s for s in start if s["id"] == b["id"])
        hp = sim.handle_point(b["type"])
        for p in (np.asarray(s["pos"]) + hp, sim.SITE + np.asarray(b["pos"]) + hp):
            if not any(w.ik(p, sim.pitch_quat(a))[1] < 0.005 for a in sim.GRASP_PITCH_DEG[:1]):
                return False
    return True


def make_scene(scene_id, tier, split, seed, variant=None):
    rng = np.random.default_rng(seed)
    while True:
        name, blocks = templates(tier, rng, variant)
        colors = rng.choice(sorted(sim.COLORS), len(blocks), replace=False)
        for i, (b, c) in enumerate(zip(blocks, colors)):
            b.update(id=f"b{i}", color=str(c))
        support_relations(blocks)
        start = sample_start(blocks, rng)
        if start and reachable(blocks, start) and sim.settle([to_world(b) for b in blocks])["stable"]:
            break
    d = SCENES / split
    d.mkdir(parents=True, exist_ok=True)
    image = sim.render_structure([to_world(b) for b in blocks], str(d / f"{scene_id}.png"))
    scene = {"id": scene_id, "tier": tier, "split": split, "seed": seed, "structure": name,
             "target": {"blocks": blocks}, "oracle_order": oracle_order(blocks), "start": start,
             "image": str(Path(image).relative_to(ROOT))}
    (d / f"{scene_id}.json").write_text(json.dumps(scene, indent=1))
    return scene


def load_scene(scene_id: str) -> dict:
    for split in ("dev", "heldout"):
        f = SCENES / split / f"{scene_id}.json"
        if f.exists():
            return json.loads(f.read_text())
    raise FileNotFoundError(f"no scene {scene_id!r}")


def generate(n_dev=6, n_heldout=4):
    """Generate all tiers once; held-out seeds are disjoint from dev seeds."""
    from lab import record

    out = []
    for t, tier in enumerate(["T0", "T1", "T2", "T3"]):
        for split, n, base in (("dev", n_dev, 1000), ("heldout", n_heldout, 9000)):
            for i in range(n):
                sid = f"{tier}-{'dev' if split == 'dev' else 'ho'}-{i + 1:02d}"
                sc = make_scene(sid, tier, split, seed=base + 100 * t + i, variant=i)
                if record.get(sid) is None:
                    record.add("scene", "team_setup", {"tier": tier, "split": split, "structure": sc["structure"],
                                                       "spec_path": f"scenes/{split}/{sid}.json", "image_path": sc["image"],
                                                       "n_blocks": len(sc["target"]["blocks"])}, obj_id=sid)
                out.append(sid)
                print(sid, sc["structure"], len(sc["target"]["blocks"]), "blocks")
    return out


if __name__ == "__main__":
    generate()
