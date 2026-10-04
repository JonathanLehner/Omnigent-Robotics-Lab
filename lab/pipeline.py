"""Assembly pipeline: the method the lab develops. Agents edit methods (configs, stage implementations,
prompts), not this contract.

A method version = methods/<name>.yaml + the git commit. The config names one implementation per stage:
  perceive    image -> structure spec            oracle | vlm
  plan_order  spec -> build order                oracle | support_sort | llm
  build       approach, grasp, carry, place, release per block (rung 0: scripted IK + weld grasp)
  verify      settle 5 s, compare to the oracle target (lab/metrics.py, frozen)
Register a new implementation with @stage("plan_order", "my_impl") in this file or in lab/stages/*.py.
"""

import importlib
import json
import pkgutil
from pathlib import Path

import numpy as np
import yaml

from lab import metrics, scenes, sim
from lab.method_models import call_model

ROOT = Path(__file__).resolve().parent.parent
STAGES: dict[str, dict] = {"perceive": {}, "plan_order": {}, "build": {}}


def stage(kind, name):
    def reg(fn):
        STAGES[kind][name] = fn
        return fn
    return reg


def load_method(name: str) -> dict:
    _load_plugins()  # the tool server is long-lived: pick up stages the engineer added since it started
    cfg = yaml.safe_load((ROOT / "methods" / f"{name}.yaml").read_text())
    for kind in STAGES:
        impl = cfg["stages"][kind]
        if impl not in STAGES[kind]:
            raise ValueError(f"method {name}: unknown {kind} implementation {impl!r}; have {sorted(STAGES[kind])}")
    return cfg


# --- perceive ----------------------------------------------------------------
@stage("perceive", "oracle")
def perceive_oracle(scene, cfg):
    return json.loads(json.dumps(scene["target"]["blocks"])), {}


@stage("perceive", "vlm")
def perceive_vlm(scene, cfg):
    """Parse the target picture with a method model. Prompt + schema are versioned files under prompts/.
    cfg perceive.image: main (single front-left picture, default) | multiview (2x2: main, top, robot side, left side)."""
    p = cfg["perceive"]
    picture = scene[{"main": "image", "multiview": "image_multiview"}[p.get("image", "main")]]
    res = call_model(p["model"], p["prompt"], text=p.get("text", ""), image=str(ROOT / picture))
    blocks = res["output"]["blocks"]
    for i, b in enumerate(blocks):
        b.setdefault("id", f"p{i}")
        b.setdefault("yaw", 0.0)
    scenes.support_relations(blocks)
    info = {"model_key": res["key"], "cached": res["cached"]}
    if cfg.get("telemetry", {}).get("schema") == "h020_combo_v1":
        info.update(
            {
                "prompt_sha256": res["prompt_sha256"],
                "image_sha256": res["image_sha256"],
                "response_sha256": res["response_sha256"],
            }
        )
    return blocks, info


# --- plan_order --------------------------------------------------------------
@stage("plan_order", "oracle")
def order_oracle(spec, scene, cfg):
    by_id = {b["id"]: b for b in spec}
    if set(by_id) == set(scene["oracle_order"]):
        return scene["oracle_order"]
    return scenes.oracle_order(spec)  # perceived ids differ: fall back to z-then-y ordering


@stage("plan_order", "support_sort")
def order_support_sort(spec, scene, cfg):
    """Symbolic: topological sort on `on` relations (supports first), ties left to right. Each prefix must
    pass the settle test, otherwise the order is rejected."""
    done, order = set(), []
    while len(order) < len(spec):
        ready = [b for b in spec if b["id"] not in done and set(b["on"]) <= done]
        if not ready:
            raise RuntimeError("order_infeasible: cyclic or unsupported spec")
        nxt = min(ready, key=lambda b: (b["pos"][2], b["pos"][1]))
        order.append(nxt["id"])
        done.add(nxt["id"])
        prefix = [scenes.to_world(b) for b in spec if b["id"] in done]
        if not sim.settle(prefix, seconds=1.5)["stable"]:
            raise RuntimeError(f"order_infeasible: prefix {order} not stable")
    return order


@stage("plan_order", "llm")
def order_llm(spec, scene, cfg):
    p = cfg["plan_order"]
    res = call_model(p["model"], p["prompt"], text=json.dumps({"blocks": spec}))
    return res["output"]["order"]


# --- build -------------------------------------------------------------------
def _match(spec, scene):
    """Perceived block -> physical block, by (color, type). Unmatched blocks are perception errors."""
    phys = {b["id"]: b for b in scene["target"]["blocks"]}
    free, pairs, failures = dict(phys), {}, []
    for b in spec:
        hit = next((pid for pid, pb in free.items() if pb["color"] == b.get("color") and pb["type"] == b.get("type")), None)
        if hit is None:
            failures.append("perception_mismatch")
            continue
        pairs[b["id"]] = hit
        free.pop(hit)
    if free:
        failures.append("perception_missing_block")
    return pairs, failures


@stage("build", "scripted_weld")
def build_scripted_weld(world, spec, order, pairs, cfg, log):
    """Rung 0: fixed base, scripted joint-space arm motion via IK, idealized weld grasp."""
    prm = cfg.get("params", {})
    seg, clear, lift = prm.get("segment_s", 1.0), prm.get("place_clearance_m", 0.005), prm.get("carry_height_m", 0.15)
    by_id = {b["id"]: b for b in spec}
    top = 0.0
    for sid in order:
        if sid not in pairs:
            continue
        b, pid = by_id[sid], pairs[sid]
        hp = sim.handle_point(b["type"])
        bp, _ = world.block_pose(pid)
        target = sim.SITE + np.asarray(b["pos"])
        pick, place = bp + hp, target + hp
        pitch = next((a for a in sim.GRASP_PITCH_DEG
                      if world.ik(pick, sim.pitch_quat(a))[1] < 0.01 and world.ik(place, sim.pitch_quat(a))[1] < 0.01), None)
        if pitch is None:
            log["failures"].append("unreachable")
            continue
        q = sim.pitch_quat(pitch)
        back = sim.rot(q, [-0.10, 0, 0])
        carry_z = max(top, target[2]) + lift
        world.set_gripper(True)
        for p in (pick + back + [0, 0, 0.05], pick + back, pick):  # approach
            world.move_joints(world.ik(p, q)[0], seg)
        world.set_gripper(False)
        world.grasp(pid)
        for p in (pick + [0, 0, lift], [place[0], place[1], carry_z], place + [0, 0, 0.04], place + [0, 0, clear]):  # carry, place
            world.move_joints(world.ik(p, q)[0], seg)
        pre = world.block_pose(pid)
        world.release()
        world.set_gripper(True)
        for p in (place + back, place + back + [0, 0, 0.1]):  # retreat
            world.move_joints(world.ik(p, q)[0], seg * 0.7)
        log["stages"].append({"block": pid, "pitch": pitch, "pre_release_err_m": float(np.linalg.norm(pre[0] - target))})
        top = max(top, target[2] + sim.BLOCK_TYPES[b["type"]][2] / 2)


# --- episode -----------------------------------------------------------------
def _pose_dict(pose):
    """JSON-ready world-frame pose for episode telemetry."""
    if pose is None:
        return None
    pos, quat = pose
    return {
        "position_xyz": np.asarray(pos, dtype=float).tolist(),
        "quaternion_wxyz": np.asarray(quat, dtype=float).tolist(),
    }


def _record_inspection_event(trace, world, kind, block_id):
    """Append an inspection event, coalescing weld and gripper-open release hooks."""
    if trace is None:
        return
    timestamp = float(world.d.time)
    if trace["events"]:
        previous = trace["events"][-1]
        if (
            previous["kind"] == kind
            and previous["block_id"] == block_id
            and previous["time_s"] == timestamp
        ):
            return
    pos, _ = world.block_pose(block_id)
    trace["events"].append(
        {
            "kind": kind,
            "block_id": block_id,
            "time_s": timestamp,
            "block_xy_m": np.asarray(pos[:2], dtype=float).tolist(),
        }
    )


def _observe_pre_release_poses(world, inspection_trace=None):
    """Record pose reads immediately preceding weld or gripper release.

    The wrappers only read simulator state and delegate to the original methods,
    so enabling episode telemetry cannot affect the physics trajectory.
    """
    poses = {}
    last_pose = [None, None]
    saw_open = False
    block_pose = world.block_pose
    release = world.release
    set_gripper = world.set_gripper
    gripper_act = world._act("arm_f1x")

    def observed_block_pose(block_id):
        pose = block_pose(block_id)
        last_pose[:] = [block_id, pose]
        return pose

    def observed_release():
        if world.held:
            block_id = world.held[0]
            poses[block_id] = block_pose(block_id)
            _record_inspection_event(
                inspection_trace, world, "release", block_id
            )
        return release()

    def observed_set_gripper(open_):
        nonlocal saw_open
        # Contact-grasp stages release by reopening the fingers rather than by
        # calling World.release(). Ignore the initial open command.
        if open_:
            is_already_open = np.isclose(world.d.ctrl[gripper_act], -1.0)
            if saw_open and not is_already_open and last_pose[0] is not None:
                poses[last_pose[0]] = last_pose[1]
                _record_inspection_event(
                    inspection_trace, world, "release", last_pose[0]
                )
            saw_open = True
        return set_gripper(open_)

    world.block_pose = observed_block_pose
    world.release = observed_release
    world.set_gripper = observed_set_gripper
    return poses


def run_episode(
    scene: dict,
    cfg: dict,
    seed: int,
    frame_every_s: float | None = None,
    inspection: bool = False,
) -> dict:
    rng = np.random.default_rng(seed)
    log = {
        "scene": scene["id"],
        "tier": scene["tier"],
        "seed": seed,
        "method": cfg["name"],
        "scene_files_sha256": dict(scene.get("_loaded_scene_files_sha256", {})),
        "failures": [],
        "stages": [],
    }
    start = {s["id"]: s for s in scene["start"]}
    blocks = [scenes.to_world(b, pos=np.asarray(start[b["id"]]["pos"]) - sim.SITE + [*rng.uniform(-0.02, 0.02, 2), 0],
                              yaw=float(rng.uniform(-3, 3)))
              for b in scene["target"]["blocks"]]
    spec, order, planner_error = [], [], None
    try:
        spec, info = STAGES["perceive"][cfg["stages"]["perceive"]](scene, cfg)
        log["perceive"] = info | {"n_blocks": len(spec)}
    except Exception as e:  # noqa: BLE001 - a broken stage is a result, not a crash
        log["failures"].append(f"stage_error:{type(e).__name__}")
    pairs, fails = _match(spec, scene)
    log["failures"] += fails
    anchored_telemetry = (
        log.get("perceive", {}).get("telemetry_schema") == "v47_anchor_v1"
    )
    if anchored_telemetry:
        from lab.stages import vlm_anchor_telemetry

        vlm_anchor_telemetry.attach_position_errors(
            log["perceive"], spec, scene, pairs
        )
    from lab.stages import h020_telemetry

    h020_telemetry_enabled = h020_telemetry.enabled(cfg)
    if h020_telemetry_enabled and "perceive" in log:
        h020_telemetry.attach_perception_errors(
            log["perceive"], spec, scene, pairs
        )
    if "perceive" in log:
        try:
            order = STAGES["plan_order"][cfg["stages"]["plan_order"]](
                spec, scene, cfg
            )
            log["order"] = order
        except Exception as e:  # noqa: BLE001 - a broken stage is a result
            planner_error = str(e)
            log["failures"].append(
                planner_error
                if "infeasible" in planner_error
                else f"stage_error:{type(e).__name__}"
            )
            log["order_error"] = planner_error
    if h020_telemetry_enabled and "perceive" in log:
        h020_telemetry.attach_order(
            log, order, spec, scene, pairs, planner_error=planner_error
        )
    pictures = {}
    if frame_every_s:  # videos show the target: ghost blocks at the site + the picture Spot was given
        views = scene.get("image_views", {})
        pictures = {"target (given)": ROOT / scene["image"], "target top": ROOT / views["top"]} if "top" in views \
            else {"target (given)": ROOT / scene["image"]}
    world = sim.World(blocks, arm_block_collision=cfg.get("arm_block_collision", False), frame_every_s=frame_every_s,
                      target=[scenes.to_world(b) for b in scene["target"]["blocks"]] if frame_every_s else None,
                      target_pictures=pictures, physics=cfg.get("physics"))
    inspection_trace = {"frames": [], "events": []} if inspection else None
    if inspection_trace is not None:
        render = world.render
        block_ids = [block["id"] for block in blocks]

        def inspected_render(*args, **kwargs):
            frame = render(*args, **kwargs)
            poses = {block_id: world.block_pose(block_id)[0] for block_id in block_ids}
            inspection_trace["frames"].append(
                {
                    "index": len(inspection_trace["frames"]),
                    "time_s": float(world.d.time),
                    "block_xy_m": {
                        block_id: pose[:2].astype(float).tolist()
                        for block_id, pose in poses.items()
                    },
                    "block_z_m": {
                        block_id: float(pose[2])
                        for block_id, pose in poses.items()
                    },
                    "active_block_id": world.held[0] if world.held else None,
                }
            )
            return frame

        world.render = inspected_render
    pre_release_poses = _observe_pre_release_poses(world, inspection_trace)
    STAGES["build"][cfg["stages"]["build"]](world, spec, order, pairs, cfg, log)
    world.step(int(metrics.SETTLE_S / world.m.opt.timestep))  # verify: settle after the last release
    per_block = []
    for b in scene["target"]["blocks"]:
        pos, quat = world.block_pose(b["id"])
        target_pos = sim.SITE + np.asarray(b["pos"])
        target = scenes.to_world(b)
        pe, ae = metrics.block_errors(target_pos, b["yaw"], pos, quat)
        per_block.append({
            "id": b["id"],
            "pos_err": pe,
            "ang_err": ae,
            "final_pose": _pose_dict((pos, quat)),
            "target_pose": _pose_dict((target["pos"], target["quat"])),
            "pre_release_pose": _pose_dict(pre_release_poses.get(b["id"])),
        })
    log["blocks"] = per_block
    log["success"] = metrics.episode_success(per_block)
    if anchored_telemetry:
        vlm_anchor_telemetry.attach_placement_residuals(
            log["perceive"], per_block, spec, pairs
        )
        if (
            not log["success"]
            and vlm_anchor_telemetry.has_localization_failure(
                log["perceive"], per_block
            )
        ):
            log["failures"].append("perception_localization")
    if h020_telemetry_enabled and "perceive" in log:
        h020_telemetry.attach_placement_residuals(
            log["perceive"], per_block, spec
        )
        attribution = h020_telemetry.classify_failure(log)
        log["failure_attribution"] = attribution
        if (
            attribution
            and attribution != "walk_drift_absorbed"
            and attribution not in log["failures"]
        ):
            log["failures"].append(attribution)
    if not log["success"] and not log["failures"]:
        log["failures"].append("placement_error")
    log["sim_time_s"] = float(world.d.time)
    if frame_every_s:
        log["_frames"] = world.frames
    if inspection:
        log["_inspection"] = inspection_trace
    return log


def _load_plugins():
    """Import new stage modules and reload changed ones (lab/stages/*.py register via @stage)."""
    import sys

    pkg = ROOT / "lab" / "stages"
    if pkg.is_dir():
        for mod in pkgutil.iter_modules([str(pkg)]):
            name = f"lab.stages.{mod.name}"
            if name in sys.modules:
                m = sys.modules[name]
                if Path(m.__file__).stat().st_mtime > getattr(m, "_loaded_mtime", 0):
                    importlib.reload(m)
            else:
                m = importlib.import_module(name)
            m._loaded_mtime = Path(m.__file__).stat().st_mtime


_load_plugins()
