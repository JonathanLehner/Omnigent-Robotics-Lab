"""H-026: deterministic footprint type override layered on v47 anchoring."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
from PIL import Image

from lab.method_models import call_model
from lab.pipeline import stage
from lab.stages.vlm_measure import _color_mask, _sample_text
from lab.stages.vlm_measure_anchor import (
    IMAGE_HEIGHT_PX,
    IMAGE_WIDTH_PX,
    PRINCIPAL_COLUMN_PX,
    TELEMETRY_SCHEMA,
    _depth_and_scale,
    _handle_candidates,
    _picture_inputs,
    measure_blocks_from_picture,
)

TYPE_TELEMETRY_SCHEMA = "h026_v47b_type_override_v1"
GEOMETRIC_CLASSES = {"cube", "brick", "ambiguous"}


def _mask_runs(row: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive true runs in one mask row."""
    xs = np.flatnonzero(row)
    if not len(xs):
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(xs[0])
    for value in xs[1:]:
        value = int(value)
        if value > previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs


def _footprint_width_px(
    image: np.ndarray,
    color: str,
    handle: tuple[float, int, int, int, int],
    mask: np.ndarray | None = None,
) -> float | None:
    """Measure the face width in an unobscured strip immediately below a handle.

    Blocks above can occlude the upper face, and the floor can contaminate rows
    near a bottom face.  The eight rows beginning two pixels below the handle
    avoid both regions in the fixed target-camera render.  A split around the
    black handle is joined before measuring.
    """
    center, _left, _top, _right, bottom = handle
    mask = _color_mask(image, color) if mask is None else mask
    widths: list[int] = []
    for row_index in range(
        min(image.shape[0], bottom + 2),
        min(image.shape[0], bottom + 10),
    ):
        runs = _mask_runs(mask[row_index])
        original_runs = list(runs)
        for left, right in zip(original_runs, original_runs[1:]):
            if (
                right[0] - left[1] - 1 <= 35
                and left[0] <= center <= right[1]
            ):
                runs.append((left[0], right[1]))
        for left, right in runs:
            width = right - left + 1
            if left - 8 <= center <= right + 8 and 20 <= width <= 140:
                widths.append(width)
    return float(np.median(widths)) if widths else None


def _expected_handle_row(layer: int) -> float:
    """Project the known handle height into the calibrated target camera."""
    # The renderer camera has zero azimuth, so structure y maps only to image u.
    # Projecting the handle-front point yields these rows; use the same calibrated
    # depth as the lateral anchor rather than inspecting simulator state.
    depth, pixels_per_m = _depth_and_scale(layer, "handle_front_center")
    del depth
    camera_elevation = math.radians(12.0)
    camera_center_x = -1.1 * math.cos(camera_elevation)
    camera_center_z = 0.12 + 1.1 * math.sin(camera_elevation)
    point_x = -0.09
    point_z = 0.05 + 0.10 * layer + 0.03
    screen_up_m = (
        (point_x - camera_center_x) * math.sin(camera_elevation)
        + (point_z - camera_center_z) * math.cos(camera_elevation)
    )
    return (IMAGE_HEIGHT_PX - 1) / 2.0 - screen_up_m * pixels_per_m


def _assign_handles(
    parsed_blocks: list[dict],
    image: np.ndarray,
) -> dict[str, tuple[tuple[float, int, int, int, int], float]]:
    """Greedily assign distinct anchored handles to parsed blocks."""
    handles = _handle_candidates(image)
    masks = {
        block["color"]: _color_mask(image, block["color"])
        for block in parsed_blocks
    }
    candidates: list[
        tuple[
            float,
            str,
            int,
            tuple[float, int, int, int, int],
            float,
        ]
    ] = []
    for block in parsed_blocks:
        layer = int(block["layer"])
        _depth, pixels_per_m = _depth_and_scale(layer, "handle_front_center")
        rough_y_m = float(block.get("rough_y_cm", 0.0)) / 100.0
        expected_column = PRINCIPAL_COLUMN_PX - rough_y_m * pixels_per_m
        expected_row = _expected_handle_row(layer)
        for handle_index, handle in enumerate(handles):
            width_px = _footprint_width_px(
                image, block["color"], handle, masks[block["color"]]
            )
            if width_px is None:
                continue
            handle_row = (handle[2] + handle[4]) / 2.0
            score = (handle[0] - expected_column) ** 2 + 4.0 * (
                handle_row - expected_row
            ) ** 2
            candidates.append(
                (score, str(block["id"]), handle_index, handle, width_px)
            )

    assignments = {}
    assigned_blocks: set[str] = set()
    assigned_handles: set[int] = set()
    for _score, block_id, handle_index, handle, width_px in sorted(candidates):
        if block_id in assigned_blocks or handle_index in assigned_handles:
            continue
        assignments[block_id] = (handle, width_px)
        assigned_blocks.add(block_id)
        assigned_handles.add(handle_index)
    return assignments


def override_types_from_picture(
    parsed_blocks: list[dict],
    measurement_picture: str | Path,
    override_cfg: dict,
) -> tuple[list[dict], list[dict]]:
    """Apply H-026's fixed 25%/75% band and return per-block telemetry."""
    image = np.asarray(Image.open(measurement_picture).convert("RGB"))
    if image.shape[:2] != (IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX):
        raise ValueError(
            f"robot-side target image must be {IMAGE_WIDTH_PX}x{IMAGE_HEIGHT_PX}, "
            f"got {image.shape[1]}x{image.shape[0]}"
        )

    blocks = [copy.deepcopy(block) for block in parsed_blocks]
    for index, block in enumerate(blocks):
        block.setdefault("id", f"p{index}")
    assignments = _assign_handles(blocks, image)

    cube_edge_m = float(override_cfg["nominal_cube_edge_m"])
    brick_length_m = float(override_cfg["nominal_brick_length_m"])
    ratio = brick_length_m / cube_edge_m
    cube_max_q = 1.0 + float(override_cfg["cube_band_fraction"]) * (ratio - 1.0)
    brick_min_q = 1.0 + float(override_cfg["brick_band_fraction"]) * (ratio - 1.0)

    rows = []
    for block in blocks:
        vlm_type = str(block["type"])
        assignment = assignments.get(str(block["id"]))
        if assignment is None:
            handle_u = width_px = measured_length_m = q = None
            geometric_class = "ambiguous"
        else:
            handle, width_px = assignment
            handle_u = float(handle[0])
            _depth, pixels_per_m = _depth_and_scale(
                int(block["layer"]), "camera_facing_face_center"
            )
            measured_length_m = float(width_px / pixels_per_m)
            q = measured_length_m / cube_edge_m
            if q <= cube_max_q:
                geometric_class = "cube"
            elif q >= brick_min_q:
                geometric_class = "brick"
            else:
                geometric_class = "ambiguous"
        if geometric_class not in GEOMETRIC_CLASSES:
            raise AssertionError(f"unexpected geometric class {geometric_class!r}")
        final_type = (
            vlm_type if geometric_class == "ambiguous" else geometric_class
        )
        override_fired = final_type != vlm_type
        block["type"] = final_type
        rows.append(
            {
                "perceived_id": block["id"],
                "vlm_type": vlm_type,
                "measured_length_m": measured_length_m,
                "q": q,
                "geometric_class": geometric_class,
                "final_type": final_type,
                "override_fired": override_fired,
                "footprint_width_px": width_px,
                "footprint_handle_u_px": handle_u,
            }
        )
    return blocks, rows


def apply_v47b_to_response(
    parsed_blocks: list[dict],
    measurement_picture: str | Path,
    cfg: dict,
) -> tuple[list[dict], dict]:
    """Apply the override and then run unchanged v47 anchored localization."""
    overridden, type_rows = override_types_from_picture(
        parsed_blocks,
        measurement_picture,
        cfg["perceive"]["type_override"],
    )
    blocks, measurement_info = measure_blocks_from_picture(
        overridden, measurement_picture
    )
    type_by_id = {row["perceived_id"]: row for row in type_rows}
    for row in measurement_info["block_measurements"]:
        row.update(type_by_id[row["perceived_id"]])
    return blocks, measurement_info | {
        "type_override_schema": TYPE_TELEMETRY_SCHEMA,
        "type_override_thresholds": {
            "nominal_cube_edge_m": float(
                cfg["perceive"]["type_override"]["nominal_cube_edge_m"]
            ),
            "nominal_brick_length_m": float(
                cfg["perceive"]["type_override"]["nominal_brick_length_m"]
            ),
            "cube_band_fraction": float(
                cfg["perceive"]["type_override"]["cube_band_fraction"]
            ),
            "brick_band_fraction": float(
                cfg["perceive"]["type_override"]["brick_band_fraction"]
            ),
        },
        "type_measurements": type_rows,
    }


@stage("perceive", "vlm_measure_anchored_type_override")
def perceive_vlm_measure_anchored_type_override(scene, cfg):
    """Run the unchanged v47 parse, then H-026's deterministic type override."""
    vlm_picture, measurement_picture = _picture_inputs(scene)
    perceive_cfg = cfg["perceive"]
    result = call_model(
        perceive_cfg["model"],
        perceive_cfg["prompt"],
        text=_sample_text(perceive_cfg),
        image=str(vlm_picture),
        use_cache=bool(perceive_cfg.get("cached", True)),
    )
    vlm_blocks = copy.deepcopy(result["output"]["blocks"])
    blocks, measurement_info = apply_v47b_to_response(
        vlm_blocks, measurement_picture, cfg
    )
    return blocks, measurement_info | {
        # Keep v47's schema so its existing localization telemetry remains active.
        "telemetry_schema": TELEMETRY_SCHEMA,
        "model_key": result["key"],
        "cached": bool(result["cached"]),
        "cache_mode": perceive_cfg.get("cache_mode", "fresh_per_seed"),
        "prompt_sha256": result["prompt_sha256"],
        "image_sha256": result["image_sha256"],
        "response_sha256": result["response_sha256"],
        "vlm_output_sha256": result["response_sha256"],
        "vlm_parsed_blocks": vlm_blocks,
        "parsed_blocks": copy.deepcopy(blocks),
    }
