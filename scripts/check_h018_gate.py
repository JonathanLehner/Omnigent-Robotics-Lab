"""Run and check H-018's pre-registered G1-G9 construct gate."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from lab import pipeline, scenes


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


def _stage(episode: dict) -> dict:
    return next(
        row
        for row in episode["stages"]
        if row["phase"] == "mpc_weld_carry_place"
    )


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
    start_z = float(scene["start"][0]["pos"][2])
    lift_threshold = start_z + 0.05
    lifted_rows = [
        row for row in before_release if row["block_pose"][2] >= lift_threshold
    ]
    first_lift_time = (
        lifted_rows[0]["time_s"] if lifted_rows else float("inf")
    )
    target_xy = np.asarray(stage["scene_object_goal_pose"][:2])
    block_type = scene["target"]["blocks"][0]["type"]
    full_xy = {
        "cube": np.array([0.12, 0.12]),
        "brick": np.array([0.12, 0.24]),
    }[block_type]
    # "Except at the target" means the carried proxy overlaps its target
    # footprint. Two identical footprints can overlap while their centres are
    # up to the sum of their circumscribed radii apart. This conservative,
    # orientation-independent geometry check does not relax release criteria.
    target_overlap_radius = float(np.linalg.norm(full_xy))
    floor_away_from_target = sum(
        row["block_floor_contacts"]
        for row in before_release
        if row["time_s"] >= first_lift_time
        and np.linalg.norm(
            np.asarray(row["block_pose"][:2]) - target_xy
        )
        > target_overlap_radius
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
        "G2": {
            "pass": bool(lifted_rows and floor_away_from_target == 0),
            "max_pre_release_z_m": max(
                (row["block_pose"][2] for row in before_release),
                default=None,
            ),
            "lift_threshold_m": lift_threshold,
            "floor_contacts_away_from_target_after_lift": (
                floor_away_from_target
            ),
            "target_overlap_radius_m": target_overlap_radius,
        },
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
            "pass": logging_ok,
            "jsonl": str(step_log_path),
            "rows": len(step_rows),
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
        checks = check_episode(
            scene, episode, base_method["idealizations"]
        )
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
