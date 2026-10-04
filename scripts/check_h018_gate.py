"""Run and check H-018's pre-registered G1-G9 construct gate."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from lab import pipeline, scenes, sim


SCENE_IDS = tuple(f"T0-dev-{index:02d}" for index in range(1, 7))
EXPECTED_TERMS = {
    "place_linear",
    "place_quadratic",
    "terminal_place",
    "place_orientation",
    "upright",
    "collision",
    "object_settle",
    "object_angular_velocity",
    "base_stability",
    "action_rate",
    "hover",
    "retreat",
    "gripper_object",
    "gripper_height",
}
class G2GeometryError(ValueError):
    """Raised when the rest-height precondition makes G2a unscorable."""

    def __init__(self, *, z_rest: float, goal_z: float):
        self.details = {
            "reason": "z_rest_goal_z_mismatch",
            "z_rest_m": z_rest,
            "goal_z_m": goal_z,
            "absolute_difference_m": abs(z_rest - goal_z),
            "tolerance_m": 0.001,
        }
        super().__init__(json.dumps(self.details, sort_keys=True))


def _stage(episode: dict) -> dict:
    return next(
        row
        for row in episode["stages"]
        if row["phase"] == "mpc_weld_carry_place"
    )


def _g2_checks(
    scene: dict,
    physics: list[dict],
    target_xy: np.ndarray,
    goal_z: float,
) -> dict[str, dict]:
    """Evaluate the binding second 2026-10-04 H-018 G2 definition."""
    release_index = next(
        (
            index
            for index, row in enumerate(physics)
            if index > 0
            and physics[index - 1]["weld_active"]
            and not row["weld_active"]
        ),
        None,
    )
    target_xy = np.asarray(target_xy, dtype=float)
    carried_type = scene["target"]["blocks"][0]["type"]
    target_type = scene["target"]["blocks"][0]["type"]
    carried_size = np.asarray(sim.BLOCK_TYPES[carried_type], dtype=float)
    target_size = np.asarray(sim.BLOCK_TYPES[target_type], dtype=float)
    carried_radius = float(
        np.linalg.norm(carried_size[:2]) / 2.0
    )
    target_radius = float(
        np.linalg.norm(target_size[:2]) / 2.0
    )
    overlap_radius = carried_radius + target_radius
    z_rest = float(carried_size[2] / 2.0)
    goal_z = float(goal_z)
    if abs(z_rest - goal_z) > 0.001:
        raise G2GeometryError(z_rest=z_rest, goal_z=goal_z)

    target_distances = [
        float(
            np.linalg.norm(
                np.asarray(row["block_pose"][:2], dtype=float) - target_xy
            )
        )
        for row in physics
    ]
    enter_index = next(
        (
            index
            for index, distance in enumerate(target_distances)
            if distance <= overlap_radius
        ),
        release_index,
    )
    transit_end_index = (
        min(enter_index, release_index)
        if enter_index is not None and release_index is not None
        else None
    )
    transit_rows = (
        physics[:transit_end_index]
        if transit_end_index is not None
        else []
    )
    clearance_threshold = z_rest + 0.05
    clearance_failures = [
        (index, row)
        for index, row in enumerate(transit_rows)
        if row["block_pose"][2] < clearance_threshold
    ]

    contact_rows: list[tuple[int, dict, float]] = []
    if release_index is not None:
        for index in range(release_index):
            row = physics[index]
            if row["block_floor_contacts"] <= 0:
                continue
            distance = target_distances[index]
            contact_rows.append((index, row, distance))
    outside_rows = [
        (index, row, distance)
        for index, row, distance in contact_rows
        if distance > overlap_radius
    ]

    floor_path_m = 0.0
    floor_net_m = 0.0
    first_contact_index = contact_rows[0][0] if contact_rows else None
    if first_contact_index is not None and release_index is not None:
        for index in range(first_contact_index, release_index):
            current = physics[index]
            following = physics[index + 1]
            if (
                current["block_floor_contacts"] > 0
                and following["block_floor_contacts"] > 0
            ):
                floor_path_m += float(
                    np.linalg.norm(
                        np.asarray(following["block_pose"][:2], dtype=float)
                        - np.asarray(
                            current["block_pose"][:2], dtype=float
                        )
                    )
                )
        floor_net_m = float(
            np.linalg.norm(
                np.asarray(
                    physics[release_index]["block_pose"][:2], dtype=float
                )
                - np.asarray(
                    physics[first_contact_index]["block_pose"][:2],
                    dtype=float,
                )
            )
        )

    g2a_pass = release_index is not None and not clearance_failures
    g2b_pass = release_index is not None and not outside_rows
    g2c_pass = release_index is not None and floor_path_m <= 0.01
    label = (
        "clean carry"
        if first_contact_index is None
        else "set-down"
        if g2b_pass and g2c_pass
        else "drag"
    )
    return {
        "G2a": {
            "pass": g2a_pass,
            "z_rest_m": z_rest,
            "goal_z_m": goal_z,
            "z_rest_goal_z_difference_m": abs(z_rest - goal_z),
            "clearance_threshold_m": clearance_threshold,
            "target_enter_time_s": (
                physics[enter_index]["time_s"]
                if enter_index is not None
                else None
            ),
            "release_time_s": (
                physics[release_index]["time_s"]
                if release_index is not None
                else None
            ),
            "transit_physics_steps": len(transit_rows),
            "minimum_transit_z_m": min(
                (row["block_pose"][2] for row in transit_rows),
                default=None,
            ),
            "clearance_failure_steps": len(clearance_failures),
        },
        "G2b": {
            "pass": g2b_pass,
            "carried_radius_m": carried_radius,
            "target_radius_m": target_radius,
            "allowed_centre_distance_m": overlap_radius,
            "pre_release_floor_contact_steps": len(contact_rows),
            "max_contact_centre_distance_m": max(
                (distance for _, _, distance in contact_rows),
                default=None,
            ),
            "outside_overlap_steps": len(outside_rows),
        },
        "G2c": {
            "pass": g2c_pass,
            "D_floor_m": floor_path_m,
            "threshold_m": 0.01,
            "net_displacement_m": floor_net_m,
            "first_contact_time_s": (
                physics[first_contact_index]["time_s"]
                if first_contact_index is not None
                else None
            ),
            "label": label,
        },
    }


def check_episode(
    scene: dict, episode: dict, expected_manifest: list[str]
) -> dict[str, dict]:
    stage = _stage(episode)
    physics = stage["physics_steps"]
    release_time = stage["release_time_s"]
    before_release = [
        row
        for row in physics
        if release_time is not None
        and row["time_s"] < release_time - 1e-12
    ]
    g2_checks = _g2_checks(
        scene,
        physics,
        stage["scene_object_goal_pose"][:2],
        stage["scene_object_goal_pose"][2],
    )
    iterations_ok = [
        row["samples"] == 128 and row["iterations_run"] == 4
        for row in stage["planning_steps"]
    ]
    budget_fraction = (
        sum(iterations_ok) / len(iterations_ok)
        if iterations_ok
        else 0.0
    )
    step_log_path = Path(stage.get("step_log_path", ""))
    step_rows = []
    if step_log_path.is_file():
        step_rows = [
            json.loads(line)
            for line in step_log_path.read_text().splitlines()
            if line
        ]
    required_fields = {
        "cost_terms",
        "block_goal_error",
        "weld_active",
        "base_xy_displacement_m",
        "base_yaw_change_deg",
        "robot_block_contacts_since_previous_plan",
        "block_floor_contacts_since_previous_plan",
        "samples",
        "iterations_run",
    }
    logging_ok = bool(step_rows) and all(
        required_fields <= set(row)
        and EXPECTED_TERMS == set(row["cost_terms"])
        and all(
            {"executed_best", "sample_mean", "sample_min"}
            == set(stats)
            for stats in row["cost_terms"].values()
        )
        for row in step_rows
    )
    physics_logging_ok = bool(physics) and all(
        {
            "time_s",
            "block_pose",
            "block_floor_contacts",
            "weld_active",
        }
        <= set(row)
        for row in physics
    )
    video = stage.get("video", {})
    rendered = bool(
        video.get("contact_sheet")
        and Path(video["contact_sheet"]).is_file()
    )
    checks = {
        "G1": {
            "pass": bool(
                physics
                and physics[0]["weld_active"]
                and before_release
                and all(row["weld_active"] for row in before_release)
                and max(
                    row["weld_position_drift_m"]
                    for row in before_release
                )
                <= 0.001
                and max(
                    row["weld_angle_drift_deg"]
                    for row in before_release
                )
                <= 1.0
            ),
            "max_position_drift_m": max(
                (
                    row["weld_position_drift_m"]
                    for row in before_release
                ),
                default=None,
            ),
            "max_angle_drift_deg": max(
                (
                    row["weld_angle_drift_deg"]
                    for row in before_release
                ),
                default=None,
            ),
        },
        **g2_checks,
        "G3": {
            "pass": bool(
                stage["collision_manifest"][
                    "all_robot_block_pairs_excluded"
                ]
                and stage["collision_manifest"][
                    "block_floor_collision_enabled"
                ]
                and sum(
                    row["robot_block_contacts"] for row in physics
                )
                == 0
            ),
            "robot_geom_count": stage["collision_manifest"][
                "robot_geom_count"
            ],
            "robot_block_contacts": sum(
                row["robot_block_contacts"] for row in physics
            ),
        },
        "G4": {
            "pass": bool(
                max(
                    row["base_xy_displacement_m"] for row in physics
                )
                <= 0.02
                and max(row["base_yaw_change_deg"] for row in physics)
                <= 2.0
            ),
            "max_base_xy_displacement_m": max(
                row["base_xy_displacement_m"] for row in physics
            ),
            "max_base_yaw_change_deg": max(
                row["base_yaw_change_deg"] for row in physics
            ),
        },
        "G5": {
            "pass": bool(
                stage["release_transitions"] == 1
                and stage["release_reason"] in ("predicate", "forced_time")
                and stage[
                    "object_pose_at_release_plus_post_release_s"
                ]
                is not None
                and stage.get(
                    "post_release_block_gripper_distance_growth_m",
                    -float("inf"),
                )
                >= 0.03
            ),
            "release_time_s": release_time,
            "forced_release": stage["forced_release"],
            "release_transitions": stage["release_transitions"],
            "distance_growth_m": stage.get(
                "post_release_block_gripper_distance_growth_m"
            ),
        },
        "G6": {
            "pass": budget_fraction >= 0.95,
            "qualified_fraction": budget_fraction,
            "planning_steps": len(iterations_ok),
        },
        "G7": {
            "pass": logging_ok and physics_logging_ok,
            "jsonl": str(step_log_path),
            "rows": len(step_rows),
            "physics_rows": len(physics),
            "physics_resolution_block_contact_log": physics_logging_ok,
        },
        "G8": {
            "pass": bool(
                len(stage["runtime_idealization_manifest"])
                == len(expected_manifest)
                and set(stage["runtime_idealization_manifest"])
                == set(expected_manifest)
                and stage["manifest_matches_yaml"]
            ),
            "runtime_manifest": stage[
                "runtime_idealization_manifest"
            ],
        },
        "G9": {
            "pass": rendered,
            "contact_sheet": video.get("contact_sheet"),
            "gif": video.get("gif"),
        },
    }
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--output-dir", default="runs/h018_pre_run_gate"
    )
    parser.add_argument(
        "--render-count",
        type=int,
        default=2,
        help="number of episodes for which to produce G9 rollout renders",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_method = pipeline.load_method("v4_mpc_weld")
    all_pass = True
    summary = []
    for index, scene_id in enumerate(SCENE_IDS):
        scene = scenes.load_scene(scene_id)
        method = copy.deepcopy(base_method)
        if index < args.render_count:
            method["sumo"]["video_dir"] = str(
                output_dir / "render_rollout" / scene_id
            )
        episode = pipeline.run_episode(scene, method, args.seed)
        episode_path = output_dir / f"{scene_id}_s{args.seed}.json"
        episode_path.write_text(
            json.dumps(episode, indent=2, sort_keys=True)
        )
        try:
            checks = check_episode(
                scene, episode, base_method["idealizations"]
            )
        except G2GeometryError as error:
            all_pass = False
            row = {
                "scene": scene_id,
                "seed": args.seed,
                "pass": False,
                "checks": {},
                "g2_scored": False,
                "g2_stop": error.details,
                "episode_json": str(episode_path),
            }
            summary.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            print(
                f"H018_GATE_STOP scene={scene_id} "
                "reason=z_rest_goal_z_mismatch",
                flush=True,
            )
            break
        if index >= args.render_count:
            checks["G9"] = {
                "pass": True,
                "not_required": True,
                "reason": "G9 pre-registers two smoke renders",
            }
        episode_pass = all(
            result["pass"] for result in checks.values()
        )
        all_pass = all_pass and episode_pass
        row = {
            "scene": scene_id,
            "seed": args.seed,
            "pass": episode_pass,
            "checks": checks,
            "episode_json": str(episode_path),
        }
        summary.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if not all(
            checks[name]["pass"] for name in ("G2a", "G2b", "G2c")
        ):
            print(
                f"H018_GATE_STOP scene={scene_id} "
                "reason=binding_G2_failure",
                flush=True,
            )
            break
    summary_path = output_dir / "gate_summary.json"
    summary_path.write_text(
        json.dumps(
            {"pass": all_pass, "episodes": summary},
            indent=2,
            sort_keys=True,
        )
    )
    print(
        "H018_GATE "
        + json.dumps(
            {
                "pass": all_pass,
                "episodes": len(summary),
                "summary": str(summary_path),
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
