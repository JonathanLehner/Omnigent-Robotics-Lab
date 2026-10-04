"""Opt-in episode telemetry and failure attribution for H-020."""

from __future__ import annotations

import numpy as np

from lab import metrics, sim

SCHEMA = "h020_combo_v1"


def enabled(cfg: dict) -> bool:
    """Return whether the method requested the H-020 logging schema."""
    return cfg.get("telemetry", {}).get("schema") == SCHEMA


def _angle_delta_deg(actual: float, target: float) -> float:
    return (float(actual) - float(target) + 180.0) % 360.0 - 180.0


def _yaw_deg(quaternion_wxyz) -> float:
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=float)
    return float(
        np.degrees(
            np.arctan2(
                2.0 * (w * z + x * y),
                1.0 - 2.0 * (y * y + z * z),
            )
        )
    )


def _error(actual_pos, actual_yaw: float, target_pos, target_yaw: float) -> dict:
    return {
        "xyz_cm": (
            (np.asarray(actual_pos, dtype=float) - np.asarray(target_pos, dtype=float))
            * 100.0
        ).tolist(),
        "yaw_deg": _angle_delta_deg(actual_yaw, target_yaw),
    }


def _layer(block: dict) -> int:
    if "layer" in block:
        return int(block["layer"])
    return round((float(block["pos"][2]) - 0.05) / 0.10)


def attach_perception_errors(
    info: dict, spec: list[dict], scene: dict, pairs: dict[str, str]
) -> None:
    """Log signed e = perceived target - true target for each true block."""
    by_perceived = {block["id"]: block for block in spec}
    perceived_for_true = {
        true_id: perceived_id for perceived_id, true_id in pairs.items()
    }
    anchored_by_true = {
        row["true_id"]: row
        for row in info.get("block_position_errors", [])
        if row.get("e_status") == "matched"
    }
    rows = []
    for true in scene["target"]["blocks"]:
        perceived_id = perceived_for_true.get(true["id"])
        if perceived_id is None:
            rows.append(
                {
                    "true_id": true["id"],
                    "perceived_id": None,
                    "color": true["color"],
                    "true_type": true["type"],
                    "perceived_type": None,
                    "true_layer": _layer(true),
                    "perceived_layer": None,
                    "e_status": "missing",
                    "e": None,
                    "p": None,
                }
            )
            continue
        perceived = by_perceived[perceived_id]
        row = {
            "true_id": true["id"],
            "perceived_id": perceived_id,
            "color": true["color"],
            "true_type": true["type"],
            "perceived_type": perceived.get("type"),
            "true_layer": _layer(true),
            "perceived_layer": _layer(perceived),
            "e_status": "matched",
            "e": _error(
                perceived["pos"],
                perceived.get("yaw", 0.0),
                true["pos"],
                true.get("yaw", 0.0),
            ),
            "p": None,
        }
        anchor = anchored_by_true.get(true["id"], {})
        row.update(
            {
                key: anchor.get(key)
                for key in (
                    "u_px",
                    "measurement_point",
                    "d_m",
                    "y_anchored",
                    "y_recentred",
                    "ppm_camera",
                    "ppm_blocks",
                )
            }
        )
        rows.append(row)
    info["block_pose_errors"] = rows
    info["spurious_perceived_blocks"] = [
        {
            "perceived_id": perceived_id,
            "color": block.get("color"),
            "type": block.get("type"),
            "layer": _layer(block),
            "perceived_position_xyz_m": block.get("pos"),
            "perceived_yaw_deg": block.get("yaw", 0.0),
        }
        for perceived_id, block in by_perceived.items()
        if perceived_id not in pairs
    ]
    info["structure_matches_true"] = (
        len(spec) == len(scene["target"]["blocks"])
        and not info["spurious_perceived_blocks"]
        and all(
            row["e_status"] == "matched"
            and row["true_type"] == row["perceived_type"]
            and row["true_layer"] == row["perceived_layer"]
            for row in rows
        )
    )


def _support_valid(order: list[str], by_id: dict[str, dict]) -> bool:
    if len(order) != len(by_id) or len(set(order)) != len(order):
        return False
    if set(order) != set(by_id):
        return False
    done = set()
    for block_id in order:
        if not set(by_id[block_id].get("on", [])) <= done:
            return False
        done.add(block_id)
    return True


def attach_order(
    log: dict,
    order: list[str],
    spec: list[dict],
    scene: dict,
    pairs: dict[str, str],
    planner_error: str | None = None,
) -> None:
    """Log perceived and true-support validity for the planned sequence."""
    perceived_by_id = {block["id"]: block for block in spec}
    true_by_id = {block["id"]: block for block in scene["target"]["blocks"]}
    true_order = [pairs[block_id] for block_id in order if block_id in pairs]
    structure_matches_true = log.get("perceive", {}).get(
        "structure_matches_true", False
    )
    log["plan_order"] = {
        "planned_ids": list(order),
        "matched_true_ids": true_order,
        "support_valid_perceived": _support_valid(order, perceived_by_id),
        "support_valid_true": (
            _support_valid(true_order, true_by_id)
            if structure_matches_true
            else "n/a_mismatch"
        ),
        "planner_error": planner_error,
    }


def attach_placement_residuals(
    info: dict,
    per_block: list[dict],
    spec: list[dict],
) -> None:
    """Log signed p = final pose - perceived target for every matched block."""
    by_perceived = {block["id"]: block for block in spec}
    final_by_true = {block["id"]: block for block in per_block}
    for row in info.get("block_pose_errors", []):
        if row["e_status"] != "matched":
            continue
        perceived = by_perceived[row["perceived_id"]]
        final_pose = final_by_true[row["true_id"]]["final_pose"]
        final_pos = np.asarray(final_pose["position_xyz"], dtype=float) - sim.SITE
        row["p"] = _error(
            final_pos,
            _yaw_deg(final_pose["quaternion_wxyz"]),
            perceived["pos"],
            perceived.get("yaw", 0.0),
        )


def _xy(error: dict | None) -> float | None:
    if error is None:
        return None
    return float(np.linalg.norm(np.asarray(error["xyz_cm"][:2], dtype=float)))


def classify_failure(log: dict) -> str | None:
    """Apply H-020's ordered, first-match failure attribution."""
    walk = log.get("walk_end", {})
    walk_outlier = (
        float(walk.get("xy_error_cm", 0.0)) > 3.0
        or abs(float(walk.get("yaw_error_deg", 0.0))) >= 10.0
    )
    if log.get("success"):
        return "walk_drift_absorbed" if walk_outlier else None

    perception = log.get("perceive", {})
    rows = perception.get("block_pose_errors", [])
    structure_mismatch = perception.get("structure_matches_true") is False
    plan = log.get("plan_order", {})
    planner_failed_or_invalid = bool(plan.get("planner_error")) or any(
        plan.get(key) is False
        for key in ("support_valid_perceived", "support_valid_true")
    )
    if not structure_mismatch and planner_failed_or_invalid:
        return "order"
    if structure_mismatch:
        return "perception_structure"

    failed_ids = {
        block["id"]
        for block in log.get("blocks", [])
        if block["pos_err"] > metrics.POS_TOL_M
        or block["ang_err"] > metrics.ANG_TOL_DEG
    }
    failed_rows = [row for row in rows if row["true_id"] in failed_ids]
    if any(
        (_xy(row.get("e")) or 0.0) > 1.5
        and _xy(row.get("p")) is not None
        and _xy(row["p"]) <= 1.5
        for row in failed_rows
    ):
        return "perception_localization"
    if walk_outlier and any(
        _xy(row.get("p")) is not None and _xy(row["p"]) > 1.5 for row in failed_rows
    ):
        return "walk"
    if any(
        _xy(row.get("e")) is not None
        and _xy(row["e"]) <= 1.5
        and _xy(row.get("p")) is not None
        and _xy(row["p"]) > 1.5
        for row in failed_rows
    ):
        return "placement"
    if failed_rows and all(
        _xy(row.get("e")) is not None
        and _xy(row["e"]) <= 1.5
        and _xy(row.get("p")) is not None
        and _xy(row["p"]) <= 1.5
        for row in failed_rows
    ):
        return "settle/orientation"
    return "unclassified"


def walk_end_telemetry(episode: dict, bridge: dict | None = None) -> dict:
    """Report Sumo's terminal approach error before and after the state bridge."""
    terminal = np.asarray(episode["terminal_base_pose"], dtype=float)
    nominal = np.asarray(episode["nominal_base_goal_pose"], dtype=float)
    return {
        "terminal_base_pose": terminal.tolist(),
        "nominal_goal_pose": nominal.tolist(),
        "xy_error_cm": float(np.linalg.norm(terminal[:2] - nominal[:2]) * 100.0),
        "yaw_error_deg": _angle_delta_deg(
            _yaw_deg(terminal[3:7]), _yaw_deg(nominal[3:7])
        ),
        "post_bridge_base_pose": (
            bridge["assembly_base_pose"] if bridge is not None else None
        ),
    }
