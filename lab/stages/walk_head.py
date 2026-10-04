"""Opt-in H-023 walk stage with mandatory walk-end telemetry."""

from __future__ import annotations

from math import atan2, degrees

import numpy as np

from lab.pipeline import build_scripted_weld, stage
from lab.stages.sumo_build import (
    _bridge_walked_base_pose,
    _episode,
    _run_sumo,
)


def _yaw_deg(quat: np.ndarray) -> float:
    quat = np.asarray(quat, dtype=float)
    quat /= np.linalg.norm(quat)
    return float(
        degrees(
            atan2(
                2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
            )
        )
    )


def _angle_delta_deg(actual: float, target: float) -> float:
    return float((actual - target + 180.0) % 360.0 - 180.0)


def walk_end_telemetry(
    episode: dict, bridge: dict | None = None
) -> dict:
    """Log every H-023 eligibility value at the arm handoff."""
    terminal = np.asarray(episode["terminal_base_pose"], dtype=float)
    nominal = np.asarray(episode["nominal_base_goal_pose"], dtype=float)
    xy_error_m = float(np.linalg.norm(terminal[:2] - nominal[:2]))
    heading_error_deg = _angle_delta_deg(
        _yaw_deg(terminal[3:7]), _yaw_deg(nominal[3:7])
    )
    base_speed_m_s = float(
        episode.get(
            "walk_end_base_speed_m_s",
            episode["terminal_base_speed_m_s"],
        )
    )
    return {
        "terminal_base_pose": terminal.tolist(),
        "nominal_goal_pose": nominal.tolist(),
        "xy_error_m": xy_error_m,
        "xy_error_cm": xy_error_m * 100.0,
        "heading_error_deg": heading_error_deg,
        "yaw_error_deg": heading_error_deg,
        "planar_base_speed_m_s": base_speed_m_s,
        "post_bridge_base_pose": (
            bridge["assembly_base_pose"] if bridge is not None else None
        ),
    }


@stage("build", "sumo_walk_head_then_scripted_weld")
def build_sumo_walk_head_then_scripted_weld(
    world, spec, order, pairs, cfg, log
):
    """Run the H-023 walk, bridge its terminal pose, then build as in v0."""
    result = _run_sumo(cfg, episode_seed=int(log["seed"]))
    episode = _episode(result)
    log["stages"].append({"phase": "walk_approach", **episode})
    log["walk_end"] = walk_end_telemetry(episode)
    if not episode["success"]:
        log["failures"].append("walk_approach_failed")
        return
    bridge = _bridge_walked_base_pose(world, episode)
    log["stages"].append(
        {"phase": "sumo_to_assembly_state_bridge", **bridge}
    )
    log["walk_end"] = walk_end_telemetry(episode, bridge)
    build_scripted_weld(world, spec, order, pairs, cfg, log)
