"""World-anchored, target-picture-only lateral measurement for H-019."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
from PIL import Image

from lab.method_models import call_model
from lab.pipeline import ROOT, stage
from lab.stages.vlm_measure import (
    BLOCK_WIDTH_M,
    _color_mask,
    _components,
    _detect_bbox,
    _sample_text,
)

# Fixed target-camera calibration. These constants deliberately live with the
# picture measurement instead of being read from simulator state at runtime.
CAMERA_VIEW = (0.0, -12.0, 1.1)  # azimuth deg, elevation deg, distance m
CAMERA_LOOKAT_STRUCTURE_M = (0.0, 0.0, 0.12)
IMAGE_WIDTH_PX = 480
IMAGE_HEIGHT_PX = 360
CAMERA_FOVY_DEG = 45.0
PRINCIPAL_COLUMN_PX = (IMAGE_WIDTH_PX - 1) / 2.0
FOCAL_LENGTH_PX = (IMAGE_HEIGHT_PX / 2.0) / math.tan(
    math.radians(CAMERA_FOVY_DEG / 2.0)
)
CAMERA_CENTER_STRUCTURE_M = np.array(
    [
        -CAMERA_VIEW[2] * math.cos(math.radians(-CAMERA_VIEW[1])),
        0.0,
        CAMERA_LOOKAT_STRUCTURE_M[2]
        + CAMERA_VIEW[2] * math.sin(math.radians(-CAMERA_VIEW[1])),
    ]
)
CAMERA_FORWARD = np.array(
    [
        math.cos(math.radians(-CAMERA_VIEW[1])),
        0.0,
        -math.sin(math.radians(-CAMERA_VIEW[1])),
    ]
)
HANDLE_FRONT_X_M = -0.09
FACE_X_M = -0.06
HANDLE_Z_OFFSET_M = 0.03
TELEMETRY_SCHEMA = "v47_anchor_v1"


def _picture_inputs(scene: dict) -> tuple[Path, Path]:
    """Return the only two scene values picture-only perception may inspect."""
    return (
        ROOT / scene["image_multiview"],
        ROOT / scene["image_views"]["robot side"],
    )


def _handle_candidates(image: np.ndarray) -> list[tuple[float, int, int, int, int]]:
    """Return isolated black handle components as (u, left, top, right, bottom)."""
    candidates = []
    for area, left, top, right, bottom in _components(image.max(axis=2) < 32):
        width, height = right - left + 1, bottom - top + 1
        if area >= 60 and 15 <= width <= 45 and 5 <= height <= 18:
            candidates.append(((left + right) / 2.0, left, top, right, bottom))
    return candidates


def _handle_center(
    image: np.ndarray,
    color: str,
    block_type: str,
    candidates: list[tuple[float, int, int, int, int]],
) -> float | None:
    """Associate a handle with its named block using nearby face-color pixels."""
    mask = _color_mask(image, color)
    half_width = 34 if block_type == "cube" else 65
    matches = []
    for centre, _left, top, _right, bottom in candidates:
        x0 = max(0, int(round(centre)) - half_width)
        x1 = min(image.shape[1], int(round(centre)) + half_width + 1)
        y0 = max(0, top - 15)
        y1 = min(image.shape[0], bottom + 29)
        support = float(mask[y0:y1, x0:x1].mean())
        matches.append((support, centre))
    if not matches:
        return None
    support, centre = max(matches)
    return float(centre) if support >= 0.20 else None


def _depth_and_scale(layer: int, measurement_point: str) -> tuple[float, float]:
    if measurement_point == "handle_front_center":
        x = HANDLE_FRONT_X_M
        z = 0.05 + 0.10 * layer + HANDLE_Z_OFFSET_M
    elif measurement_point == "camera_facing_face_center":
        x = FACE_X_M
        z = 0.05 + 0.10 * layer
    else:
        raise ValueError(f"unknown measurement point {measurement_point!r}")
    point = np.array([x, 0.0, z])
    depth = float(np.dot(point - CAMERA_CENTER_STRUCTURE_M, CAMERA_FORWARD))
    return depth, FOCAL_LENGTH_PX / depth


def measure_blocks_from_picture(
    parsed_blocks: list[dict], measurement_picture: str | Path
) -> tuple[list[dict], dict]:
    """Attach anchored positions using only a parsed structure and picture pixels."""
    image = np.asarray(Image.open(measurement_picture).convert("RGB"))
    if image.shape[:2] != (IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX):
        raise ValueError(
            f"robot-side target image must be {IMAGE_WIDTH_PX}x{IMAGE_HEIGHT_PX}, "
            f"got {image.shape[1]}x{image.shape[0]}"
        )

    parsed_blocks = [dict(block) for block in parsed_blocks]
    for index, block in enumerate(parsed_blocks):
        block.setdefault("id", f"p{index}")

    handles = _handle_candidates(image)
    face_detections = {
        block["id"]: _detect_bbox(image, block["color"], block["type"])
        for block in parsed_blocks
    }
    measurements = {}
    for block in parsed_blocks:
        handle_u = _handle_center(image, block["color"], block["type"], handles)
        face = face_detections[block["id"]]
        if handle_u is not None:
            measurements[block["id"]] = {
                "u_px": handle_u,
                "measurement_point": "handle_front_center",
            }
        elif face is not None:
            measurements[block["id"]] = {
                "u_px": float(face[0]),
                "measurement_point": "camera_facing_face_center",
            }
        else:
            measurements[block["id"]] = {
                "u_px": None,
                "measurement_point": "vlm_rough_y",
            }

    block_scales = [
        detection[1] / BLOCK_WIDTH_M[block["type"]]
        for block in parsed_blocks
        if (detection := face_detections[block["id"]]) is not None
        and np.isfinite(detection[1])
    ]
    ppm_blocks = float(np.median(block_scales)) if block_scales else None
    bottom_columns = [
        measurements[block["id"]]["u_px"]
        for block in parsed_blocks
        if int(block["layer"]) == 0
        and measurements[block["id"]]["u_px"] is not None
    ]
    bottom_reference = (
        (min(bottom_columns) + max(bottom_columns)) / 2.0
        if bottom_columns
        else None
    )

    blocks = []
    rows = []
    for block in parsed_blocks:
        measurement = measurements[block["id"]]
        u_px = measurement["u_px"]
        fallback = u_px is None
        depth = ppm_camera = anchored_y = None
        if fallback:
            lateral_y = float(block["rough_y_cm"]) / 100.0
        else:
            depth, ppm_camera = _depth_and_scale(
                int(block["layer"]), measurement["measurement_point"]
            )
            anchored_y = (PRINCIPAL_COLUMN_PX - u_px) / ppm_camera
            lateral_y = anchored_y
        recentred_y = (
            -(u_px - bottom_reference) / ppm_blocks
            if u_px is not None
            and bottom_reference is not None
            and ppm_blocks is not None
            else None
        )
        block["pos"] = [
            0.0,
            float(lateral_y),
            0.05 + 0.10 * int(block["layer"]),
        ]
        block["yaw"] = 0.0
        block["on"] = list(block["supported_by"])
        block["lateral_measurement_fallback"] = fallback
        block["lateral_measurement_source"] = (
            "vlm_rough_y" if fallback else "target_pixels_camera_anchor"
        )
        blocks.append(block)
        rows.append(
            {
                "perceived_id": block["id"],
                "u_px": u_px,
                "measurement_point": measurement["measurement_point"],
                "d_m": depth,
                "y_anchored": anchored_y,
                "y_recentred": recentred_y,
                "ppm_camera": ppm_camera,
                "ppm_blocks": ppm_blocks,
            }
        )
    return blocks, {
        "measurement_view": "robot side",
        "camera_view": list(CAMERA_VIEW),
        "camera_lookat_structure_m": list(CAMERA_LOOKAT_STRUCTURE_M),
        "camera_fovy_deg": CAMERA_FOVY_DEG,
        "image_size_px": [IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX],
        "pixels_per_m": ppm_blocks,
        "ppm_blocks": ppm_blocks,
        "bottom_reference_px": bottom_reference,
        "fallback_blocks": sum(block["lateral_measurement_fallback"] for block in blocks),
        "block_measurements": rows,
    }


@stage("perceive", "vlm_measure_anchored")
def perceive_vlm_measure_anchored(scene, cfg):
    """Parse and measure using only the target-picture paths in ``scene``."""
    vlm_picture, measurement_picture = _picture_inputs(scene)
    perceive_cfg = cfg["perceive"]
    result = call_model(
        perceive_cfg["model"],
        perceive_cfg["prompt"],
        text=_sample_text(perceive_cfg),
        image=str(vlm_picture),
        use_cache=bool(perceive_cfg.get("cached", True)),
    )
    blocks, measurement_info = measure_blocks_from_picture(
        result["output"]["blocks"], measurement_picture
    )
    return blocks, measurement_info | {
        "telemetry_schema": TELEMETRY_SCHEMA,
        "model_key": result["key"],
        "cached": bool(result["cached"]),
        "cache_mode": perceive_cfg.get("cache_mode", "fresh_per_seed"),
        "prompt_sha256": result["prompt_sha256"],
        "image_sha256": result["image_sha256"],
        "response_sha256": result["response_sha256"],
        "vlm_output_sha256": result["response_sha256"],
        "parsed_blocks": copy.deepcopy(blocks),
    }


@stage("perceive", "logged_parse")
def perceive_logged_parse(_scene, cfg):
    """Replay a parse stored in an episode, without another model query."""
    blocks = copy.deepcopy(cfg["perceive"]["logged_blocks"])
    return blocks, {
        "replayed_logged_parse": True,
        "cached": False,
        "cache_mode": "logged_parse",
        "parsed_blocks": copy.deepcopy(blocks),
    }
