"""Pipeline build stages backed by Sumo's Relic whole-body MPC."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

from lab import sim
from lab.pipeline import build_scripted_weld, stage

ROOT = Path(__file__).resolve().parents[2]
SUMO = Path.home() / "src" / "sumo"
PIXI = Path.home() / ".pixi" / "bin" / "pixi"


def _walk_start_config(scfg: dict, episode_seed: int) -> tuple[dict, dict | None]:
    """Resolve a versioned per-seed walk start into task config fields."""
    table_name = scfg.get("walk_starts")
    if not table_name:
        return {}, None
    table_path = ROOT / "walk_starts" / f"{table_name}.yaml"
    table = yaml.safe_load(table_path.read_text())
    row = table.get("starts", {}).get(str(episode_seed))
    if row is None:
        return {}, None
    yaw_deg = float(row["yaw_deg"])
    return (
        {
            "start_x": float(row["x"]),
            "start_y": float(row["y"]),
            "start_yaw": float(np.radians(yaw_deg)),
        },
        {
            "table": table_name,
            "seed": episode_seed,
            **row,
        },
    )


def _run_sumo(
    cfg: dict,
    episode_seed: int = 0,
    object_start_pose: np.ndarray | None = None,
    object_goal_pose: np.ndarray | None = None,
    object_size: tuple[float, float, float] | None = None,
    object_mass: float | None = None,
) -> dict:
    """Run one configured Sumo episode and return the adapter's JSON result."""
    scfg = cfg["sumo"]
    task_config = dict(scfg.get("task_config", {}))
    for key in ("forced_release_s", "post_release_s"):
        if key in scfg:
            task_config.setdefault(key, scfg[key])
    start_config, walk_start = _walk_start_config(scfg, episode_seed)
    task_config.update(start_config)
    cmd = [
        str(PIXI),
        "run",
        "--manifest-path",
        str(SUMO / "pyproject.toml"),
        "python",
        str(ROOT / "lab" / "sumo_adapter" / "run_task.py"),
        "--task",
        scfg["task"],
        "--task-module",
        str((ROOT / scfg["task_module"]).resolve()),
        "--episodes",
        "1",
        "--episode-seed",
        str(episode_seed),
        "--episode-length-s",
        str(scfg["episode_length_s"]),
        "--optimizer",
        scfg.get("optimizer", "cem"),
        "--num-rollouts",
        str(scfg.get("num_rollouts", 24)),
    ]
    if "max_opt_iters_per_step" in scfg:
        cmd.extend(
            [
                "--max-opt-iters-per-step",
                str(scfg["max_opt_iters_per_step"]),
            ]
        )
    if scfg.get("log_per_step_cost"):
        log_dir = ROOT / "runs" / "sumo_step_logs"
        step_log = (
            log_dir
            / f"{scfg['task']}_{time.time_ns()}_s{episode_seed}.jsonl"
        )
        cmd.extend(["--step-log-path", str(step_log)])
    if scfg.get("video_dir"):
        cmd.extend(["--video-dir", str(Path(scfg["video_dir"]).resolve())])
    if task_config:
        cmd.extend(
            [
                "--task-config-json",
                json.dumps(task_config, separators=(",", ":")),
            ]
        )
    settle_phase = scfg.get("settle_phase")
    if settle_phase:
        cmd.extend(
            [
                "--settle-phase-s",
                str(settle_phase["duration_s"]),
                "--settle-config-json",
                json.dumps(
                    settle_phase["task_config"], separators=(",", ":")
                ),
            ]
        )
    if "horizon_s" in scfg:
        cmd.extend(["--horizon", str(scfg["horizon_s"])])
    if object_start_pose is not None:
        cmd.extend(
            [
                "--object-start-pose",
                *[str(value) for value in object_start_pose],
            ]
        )
    if object_goal_pose is not None:
        cmd.extend(
            [
                "--object-goal-pose",
                *[str(value) for value in object_goal_pose],
            ]
        )
    if object_size is not None:
        cmd.extend(["--object-size", *[str(value) for value in object_size]])
    if object_mass is not None:
        cmd.extend(["--object-mass", str(object_mass)])
    res = subprocess.run(
        cmd,
        cwd=SUMO,
        capture_output=True,
        text=True,
        timeout=float(scfg.get("timeout_s", 600)),
    )
    rows = [line for line in res.stdout.splitlines() if line.startswith("{")]
    if res.returncode or not rows:
        detail = (res.stderr or res.stdout)[-1200:].replace("\n", " ")
        raise RuntimeError(f"sumo_task_failed:{detail}")
    result = json.loads(rows[-1])
    result["walk_start"] = walk_start
    return result


def _episode(result: dict) -> dict:
    episode = result["episodes"][0]
    return {
        "sumo_run_id": result["run_id"],
        "sumo_deterministic": result["sumo_deterministic"],
        "rollout_cutoff_mode": result["rollout_cutoff_mode"],
        "eigen_threads": result["eigen_threads"],
        "onnx_inter_op_threads": result["onnx_inter_op_threads"],
        "task": result["task"],
        "optimizer": result["optimizer"],
        "rollout_backend": result["backend"],
        "num_rollouts": result["num_rollouts"],
        "controller_horizon_s": result.get("controller_horizon_s"),
        "task_config": result.get("task_config"),
        "settle_phase_s": result.get("settle_phase_s", 0.0),
        "settle_config": result.get("settle_config"),
        "phase_cost_weights": result.get("phase_cost_weights"),
        "walk_start": result.get("walk_start"),
        "object_proxy": result.get("object_proxy"),
        "object_start_pose": result.get("object_start_pose"),
        "robot_start_pose": result.get("robot_start_pose"),
        "object_goal_pose": result.get("object_goal_pose"),
        **episode,
    }


def _quat_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Sign-invariant angular distance between wxyz quaternions."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(
        np.degrees(2.0 * np.arccos(np.clip(abs(np.dot(a, b)), 0.0, 1.0)))
    )


def _log_next_settle_drift(world, block_id: str, stage_log: dict) -> None:
    """Measure the pipeline's next (verification) settle without adding a step."""
    before_pos, before_quat = world.block_pose(block_id)
    original_step = world.step

    def step_and_log(n):
        original_step(n)
        after_pos, after_quat = world.block_pose(block_id)
        drift = after_pos - before_pos
        stage_log["post_transfer_settle_drift_xyz_m"] = drift.tolist()
        stage_log["post_transfer_settle_drift_m"] = float(
            np.linalg.norm(drift)
        )
        stage_log["post_transfer_settle_drift_deg"] = _quat_error_deg(
            after_quat, before_quat
        )
        world.step = original_step

    world.step = step_and_log


def _bridge_walked_base_pose(world, episode: dict) -> dict:
    """Transfer the true terminal Sumo base pose into the fixed-root assembly frame."""
    actual = np.asarray(episode["terminal_base_pose"], dtype=float)
    nominal = np.asarray(episode["nominal_base_goal_pose"], dtype=float)
    assembly_nominal_pos, assembly_nominal_quat = world.base_pose()
    residual_quat = sim.quat_mul(sim.quat_conj(nominal[3:7]), actual[3:7])
    residual_pos = sim.rot(sim.quat_conj(nominal[3:7]), actual[:3] - nominal[:3])
    assembly_actual_pos = assembly_nominal_pos + sim.rot(assembly_nominal_quat, residual_pos)
    assembly_actual_quat = sim.quat_mul(assembly_nominal_quat, residual_quat)
    world.set_base_pose(assembly_actual_pos, assembly_actual_quat)
    return {
        "sumo_terminal_base_pose": actual.tolist(),
        "sumo_nominal_base_goal_pose": nominal.tolist(),
        "assembly_base_pose": [
            *assembly_actual_pos.tolist(),
            *assembly_actual_quat.tolist(),
        ],
    }


@stage("build", "sumo_walk_then_scripted_weld")
def build_sumo_walk_then_scripted_weld(world, spec, order, pairs, cfg, log):
    """Walk to the site with Relic, then execute the v0 weld placement."""
    result = _run_sumo(cfg, episode_seed=int(log["seed"]))
    episode = _episode(result)
    log["stages"].append({"phase": "walk_approach", **episode})
    h020_logging = (
        cfg.get("telemetry", {}).get("schema") == "h020_combo_v1"
    )
    if h020_logging:
        from lab.stages.h020_telemetry import walk_end_telemetry

        log["walk_end"] = walk_end_telemetry(episode)
    if not episode["success"]:
        log["failures"].append("walk_approach_failed")
        return
    bridge = _bridge_walked_base_pose(world, episode)
    log["stages"].append(
        {"phase": "sumo_to_assembly_state_bridge", **bridge}
    )
    if h020_logging:
        log["walk_end"] = walk_end_telemetry(episode, bridge)
    build_scripted_weld(world, spec, order, pairs, cfg, log)


@stage("build", "sumo_mpc_weld_place")
def build_sumo_mpc_weld_place(world, spec, order, pairs, cfg, log):
    """Use Sumo MPC for placement and transfer its terminal pose to the lab world.

    T0 has one block.  The Sumo task decides whether and where the block is
    placed. The Sumo proxy uses that scene block's geometry, mass, and target;
    the bridge copies its terminal free-joint pose so the frozen lab metric can
    assess the result. Picking remains an ideal weld grasp at this rung.
    """
    by_id = {block["id"]: block for block in spec}
    for sid in order:
        if sid not in pairs:
            continue
        block = by_id[sid]
        target_pos = sim.SITE + np.asarray(block["pos"], dtype=float)
        target_yaw = np.radians(float(block.get("yaw", 0.0))) / 2.0
        target_quat = np.array(
            [np.cos(target_yaw), 0.0, 0.0, np.sin(target_yaw)]
        )
        object_goal_pose = np.concatenate([target_pos, target_quat])
        pid = pairs[sid]
        start_pos, start_quat = world.block_pose(pid)
        object_start_pose = np.concatenate([start_pos, start_quat])
        result = _run_sumo(
            cfg,
            episode_seed=int(log["seed"]),
            object_start_pose=object_start_pose,
            object_goal_pose=object_goal_pose,
            object_size=sim.BLOCK_TYPES[block["type"]],
            object_mass=sim.BLOCK_MASS[block["type"]],
        )
        episode = _episode(result)
        stage_log = {
            "phase": "mpc_place",
            "block": pid,
            "assembly_bridge": "teleport_sumo_terminal_free_joint_pose",
            "bridge_snaps_to_goal": False,
            "scene_object_start_pose": object_start_pose.tolist(),
            "scene_object_goal_pose": object_goal_pose.tolist(),
            "handoff_displacement_xyz_m": None,
            "handoff_displacement_m": None,
            "handoff_displacement_deg": None,
            "post_transfer_settle_drift_xyz_m": None,
            "post_transfer_settle_drift_m": None,
            "post_transfer_settle_drift_deg": None,
            **episode,
        }
        log["stages"].append(stage_log)
        echoed_goal = episode.get("object_goal_pose")
        if echoed_goal is None or not np.allclose(
            np.asarray(echoed_goal, dtype=float),
            object_goal_pose,
            atol=1e-9,
            rtol=0.0,
        ):
            log["failures"].append("sumo_object_goal_mismatch")
            continue
        if not episode["success"]:
            log["failures"].append("mpc_place_failed")
        if episode.get("final_qpos") is None:
            continue

        scfg = cfg["sumo"]
        qpos = np.asarray(episode["final_qpos"], dtype=float)
        pose_idx = int(scfg["object_pose_qpos_index"])
        terminal_pose = qpos[pose_idx : pose_idx + 7].copy()
        joint = world.m.joint(f"{pid}_free")
        qadr = joint.qposadr[0]
        dadr = joint.dofadr[0]
        world.d.qpos[qadr : qadr + 7] = terminal_pose
        world.d.qvel[dadr : dadr + 6] = 0.0
        assembly_pos, assembly_quat = world.block_pose(pid)
        handoff_displacement = assembly_pos - terminal_pose[:3]
        stage_log.update(
            {
                "sumo_terminal_block_pose": terminal_pose.tolist(),
                "assembly_block_pose_after_transfer": [
                    *assembly_pos.tolist(),
                    *assembly_quat.tolist(),
                ],
                "handoff_displacement_xyz_m": handoff_displacement.tolist(),
                "handoff_displacement_m": float(
                    np.linalg.norm(handoff_displacement)
                ),
                "handoff_displacement_deg": _quat_error_deg(
                    assembly_quat, terminal_pose[3:7]
                ),
            }
        )
        _log_next_settle_drift(world, pid, stage_log)


@stage("build", "sumo_mpc_weld_carry_place")
def build_sumo_mpc_weld_carry_place(world, spec, order, pairs, cfg, log):
    """Run H-018's verified weld-carry MPC and bridge the post-release pose."""
    by_id = {block["id"]: block for block in spec}
    expected_manifest = cfg["idealizations"]
    for sid in order:
        if sid not in pairs:
            continue
        block = by_id[sid]
        pid = pairs[sid]
        target_pos = sim.SITE + np.asarray(block["pos"], dtype=float)
        target_yaw = np.radians(float(block.get("yaw", 0.0))) / 2.0
        target_quat = np.array(
            [np.cos(target_yaw), 0.0, 0.0, np.sin(target_yaw)]
        )
        object_goal_pose = np.concatenate([target_pos, target_quat])
        start_pos, start_quat = world.block_pose(pid)
        object_start_pose = np.concatenate([start_pos, start_quat])
        result = _run_sumo(
            cfg,
            episode_seed=int(log["seed"]),
            object_start_pose=object_start_pose,
            object_goal_pose=object_goal_pose,
            object_size=sim.BLOCK_TYPES[block["type"]],
            object_mass=sim.BLOCK_MASS[block["type"]],
        )
        episode = _episode(result)
        physics_steps = episode.get("physics_steps", [])
        pre_release = [
            row
            for row in physics_steps
            if episode.get("release_time_s") is None
            or row["time_s"] < episode["release_time_s"] - 1e-12
        ]
        max_weld_position_drift = max(
            (
                row["weld_position_drift_m"]
                for row in pre_release
            ),
            default=float("inf"),
        )
        max_weld_angle_drift = max(
            (
                row["weld_angle_drift_deg"]
                for row in pre_release
            ),
            default=float("inf"),
        )
        total_robot_block_contacts = sum(
            row["robot_block_contacts"] for row in physics_steps
        )
        max_base_xy_displacement = max(
            (
                row["base_xy_displacement_m"]
                for row in physics_steps
            ),
            default=float("inf"),
        )
        max_base_yaw_change = max(
            (row["base_yaw_change_deg"] for row in physics_steps),
            default=float("inf"),
        )
        runtime_manifest = episode.get(
            "runtime_idealization_manifest", []
        )
        manifest_matches = (
            len(runtime_manifest) == len(expected_manifest)
            and set(runtime_manifest) == set(expected_manifest)
        )
        weld_ok = bool(
            physics_steps
            and physics_steps[0]["weld_active"]
            and all(row["weld_active"] for row in pre_release)
            and max_weld_position_drift <= 0.001
            and max_weld_angle_drift <= 1.0
            and episode.get("release_transitions") == 1
        )
        collision_manifest = episode.get("collision_manifest", {})
        collision_ok = bool(
            collision_manifest.get("all_robot_block_pairs_excluded")
            and collision_manifest.get("block_floor_collision_enabled")
            and total_robot_block_contacts == 0
        )
        base_ok = bool(
            max_base_xy_displacement <= 0.02
            and max_base_yaw_change <= 2.0
        )
        tags = []
        if episode.get("forced_release"):
            tags.append("mpc_carry_forced_release")
        for ok, tag in (
            (weld_ok, "construct_weld"),
            (collision_ok, "construct_collision"),
            (base_ok, "construct_base"),
            (manifest_matches, "construct_manifest"),
        ):
            if not ok:
                tags.append(tag)
                log["failures"].append(tag)
        stage_log = {
            "phase": "mpc_weld_carry_place",
            "block": pid,
            "assembly_bridge": (
                "teleport_sumo_release_plus_1s_pose"
            ),
            "bridge_snaps_to_goal": False,
            "scene_object_start_pose": object_start_pose.tolist(),
            "scene_object_goal_pose": object_goal_pose.tolist(),
            "runtime_idealization_manifest": runtime_manifest,
            "expected_idealizations": expected_manifest,
            "manifest_matches_yaml": manifest_matches,
            "idealization_evidence": {
                "weld_active_at_t0": bool(
                    physics_steps
                    and physics_steps[0]["weld_active"]
                ),
                "max_pre_release_weld_position_drift_m": (
                    max_weld_position_drift
                ),
                "max_pre_release_weld_angle_drift_deg": (
                    max_weld_angle_drift
                ),
                "collision_masks": collision_manifest,
                "base_commands_clamped": True,
                "max_base_xy_displacement_m": max_base_xy_displacement,
                "max_base_yaw_change_deg": max_base_yaw_change,
                "bridge_mode": "release_plus_post_release_s",
                "pick_mode": "welded_at_reset",
                "release_mode": "scripted_predicate_or_forced_time",
            },
            "tags": tags,
            "construct_weld": weld_ok,
            "construct_collision": collision_ok,
            "construct_base": base_ok,
            "construct_manifest": manifest_matches,
            "handoff_displacement_xyz_m": None,
            "handoff_displacement_m": None,
            "handoff_displacement_deg": None,
            "post_transfer_settle_drift_xyz_m": None,
            "post_transfer_settle_drift_m": None,
            "post_transfer_settle_drift_deg": None,
            **episode,
        }
        log["stages"].append(stage_log)

        echoed_goal = episode.get("object_goal_pose")
        if echoed_goal is None or not np.allclose(
            np.asarray(echoed_goal, dtype=float),
            object_goal_pose,
            atol=1e-9,
            rtol=0.0,
        ):
            log["failures"].append("sumo_object_goal_mismatch")
            continue
        reported_pose_idx = episode.get("object_pose_qpos_index")
        configured_pose_idx = int(
            cfg["sumo"]["object_pose_qpos_index"]
        )
        if reported_pose_idx != configured_pose_idx:
            log["failures"].append(
                "sumo_object_pose_qpos_index_mismatch"
            )
            if "construct_manifest" not in tags:
                tags.append("construct_manifest")
            stage_log["construct_manifest"] = False
            continue
        bridge_pose = episode.get(
            "object_pose_at_release_plus_post_release_s"
        )
        if bridge_pose is None:
            log["failures"].append("mpc_carry_no_post_release_pose")
            continue
        terminal_pose = np.asarray(bridge_pose, dtype=float)
        joint = world.m.joint(f"{pid}_free")
        qadr = joint.qposadr[0]
        dadr = joint.dofadr[0]
        world.d.qpos[qadr : qadr + 7] = terminal_pose
        world.d.qvel[dadr : dadr + 6] = 0.0
        assembly_pos, assembly_quat = world.block_pose(pid)
        handoff_displacement = assembly_pos - terminal_pose[:3]
        stage_log.update(
            {
                "sumo_bridge_block_pose": terminal_pose.tolist(),
                "assembly_block_pose_after_transfer": [
                    *assembly_pos.tolist(),
                    *assembly_quat.tolist(),
                ],
                "handoff_displacement_xyz_m": (
                    handoff_displacement.tolist()
                ),
                "handoff_displacement_m": float(
                    np.linalg.norm(handoff_displacement)
                ),
                "handoff_displacement_deg": _quat_error_deg(
                    assembly_quat, terminal_pose[3:7]
                ),
            }
        )
        _log_next_settle_drift(world, pid, stage_log)
