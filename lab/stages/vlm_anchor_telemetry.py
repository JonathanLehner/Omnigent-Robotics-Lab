"""Post-perception oracle telemetry and failure classification for H-019."""

from __future__ import annotations


def attach_position_errors(
    info: dict, spec: list[dict], scene: dict, pairs: dict[str, str]
) -> None:
    """Use the runner's frozen matches to append signed e telemetry in cm."""
    by_perceived = {block["id"]: block for block in spec}
    measurement_by_id = {
        row["perceived_id"]: row for row in info.get("block_measurements", [])
    }
    perceived_for_true = {true_id: perceived_id for perceived_id, true_id in pairs.items()}
    rows = []
    for true in scene["target"]["blocks"]:
        perceived_id = perceived_for_true.get(true["id"])
        if perceived_id is None:
            rows.append(
                {
                    "true_id": true["id"],
                    "perceived_id": None,
                    "color": true["color"],
                    "type": true["type"],
                    "e_status": "missing",
                    "e": None,
                    "p": None,
                    "u_px": None,
                    "measurement_point": None,
                    "d_m": None,
                    "y_anchored": None,
                    "y_recentred": None,
                    "ppm_camera": None,
                    "ppm_blocks": info.get("ppm_blocks"),
                }
            )
            continue
        perceived = by_perceived[perceived_id]
        measurement = measurement_by_id.get(perceived_id, {})
        rows.append(
            {
                "true_id": true["id"],
                "perceived_id": perceived_id,
                "color": true["color"],
                "type": true["type"],
                "e_status": "matched",
                "e": (float(perceived["pos"][1]) - float(true["pos"][1])) * 100.0,
                "p": None,
                **{
                    key: measurement.get(key)
                    for key in (
                        "u_px",
                        "measurement_point",
                        "d_m",
                        "y_anchored",
                        "y_recentred",
                        "ppm_camera",
                        "ppm_blocks",
                    )
                },
            }
        )
    spurious = [
        {
            "perceived_id": perceived_id,
            "color": by_perceived[perceived_id].get("color"),
            "type": by_perceived[perceived_id].get("type"),
            "perceived_position_xyz_m": by_perceived[perceived_id].get("pos"),
        }
        for perceived_id in by_perceived
        if perceived_id not in pairs
    ]
    info["block_position_errors"] = rows
    info["spurious_perceived_blocks"] = spurious
    info["n_missing"] = sum(row["e_status"] == "missing" for row in rows)
    info["n_spurious"] = len(spurious)
    # Keep the raw measurements only inside the richer, matched rows.
    info.pop("block_measurements", None)


def attach_placement_residuals(
    info: dict, per_block: list[dict], spec: list[dict], pairs: dict[str, str]
) -> None:
    """Append p = final y - perceived target y in cm for every match."""
    by_perceived = {block["id"]: block for block in spec}
    final_by_true = {block["id"]: block for block in per_block}
    for row in info.get("block_position_errors", []):
        if row["e_status"] != "matched":
            continue
        final = final_by_true[row["true_id"]]["final_pose"]["position_xyz"]
        perceived_y = float(by_perceived[row["perceived_id"]]["pos"][1])
        row["p"] = (float(final[1]) - perceived_y) * 100.0


def has_localization_failure(info: dict, per_block: list[dict]) -> bool:
    """H-019 classifier: failed match, |e| > 1.5 cm, |p| <= 1.5 cm."""
    failed_true_ids = {
        block["id"]
        for block in per_block
        if block["pos_err"] > 0.03 or block["ang_err"] > 10.0
    }
    return any(
        row["true_id"] in failed_true_ids
        and row["e_status"] == "matched"
        and abs(row["e"]) > 1.5
        and row["p"] is not None
        and abs(row["p"]) <= 1.5
        for row in info.get("block_position_errors", [])
    )
