"""Post-inference position-error telemetry for the v46 rerun."""

from __future__ import annotations

import numpy as np

from lab.pipeline import stage
from lab.stages.vlm_measure import perceive_vlm_measure


def _position_errors(blocks: list[dict], scene: dict) -> list[dict]:
    """Match by appearance and log e = perceived - true in structure frame."""
    unmatched = list(scene["target"]["blocks"])
    telemetry = []
    for perceived in blocks:
        match_index = next(
            (
                index
                for index, true in enumerate(unmatched)
                if true["color"] == perceived.get("color")
                and true["type"] == perceived.get("type")
            ),
            None,
        )
        if match_index is None:
            telemetry.append(
                {
                    "perceived_id": perceived["id"],
                    "true_id": None,
                    "perceived_position_xyz_m": perceived["pos"],
                    "true_position_xyz_m": None,
                    "error_perceived_minus_true_xyz_m": None,
                }
            )
            continue
        true = unmatched.pop(match_index)
        perceived_pos = np.asarray(perceived["pos"], dtype=float)
        true_pos = np.asarray(true["pos"], dtype=float)
        telemetry.append(
            {
                "perceived_id": perceived["id"],
                "true_id": true["id"],
                "color": perceived["color"],
                "type": perceived["type"],
                "perceived_position_xyz_m": perceived_pos.tolist(),
                "true_position_xyz_m": true_pos.tolist(),
                "error_perceived_minus_true_xyz_m": (
                    perceived_pos - true_pos
                ).tolist(),
            }
        )
    return telemetry


@stage("perceive", "vlm_measure_telemetry")
def perceive_vlm_measure_telemetry(scene, cfg):
    """Run picture-only perception, then attach oracle comparison to its log."""
    blocks, info = perceive_vlm_measure(scene, cfg)
    return blocks, info | {"block_position_errors": _position_errors(blocks, scene)}
