"""Picture-only structure parsing and lateral block measurement for H-013."""

from __future__ import annotations

import colorsys
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from lab.method_models import call_model
from lab.pipeline import ROOT, stage

# These are task definitions, not simulator observations: full lateral widths in metres
# and the eight nominal render colors named in the perception prompt.
BLOCK_WIDTH_M = {"cube": 0.12, "brick": 0.24}
COLOR_RGB = {
    "red": (0.85, 0.15, 0.15),
    "blue": (0.15, 0.35, 0.85),
    "green": (0.20, 0.70, 0.25),
    "yellow": (0.95, 0.80, 0.10),
    "orange": (0.95, 0.50, 0.10),
    "purple": (0.55, 0.25, 0.75),
    "cyan": (0.10, 0.75, 0.80),
    "white": (0.92, 0.92, 0.92),
}


def _picture_inputs(scene: dict) -> tuple[Path, Path]:
    """Return the only two scene inputs this stage is allowed to inspect."""
    vlm_picture = ROOT / scene["image_multiview"]
    measurement_picture = ROOT / scene["image_views"]["robot side"]
    return vlm_picture, measurement_picture


def _hue(rgb: tuple[float, float, float]) -> float:
    return colorsys.rgb_to_hsv(*rgb)[0]


def _color_mask(image: np.ndarray, color: str) -> np.ndarray:
    """Classify a named rendered color while remaining invariant to face shading."""
    rgb = image.astype(float) / 255.0
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    delta = maximum - minimum
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 0)
    if color == "white":
        # The white block's vertical face is neutral mid-grey. The checkerboard is
        # substantially brighter; black background and handles are darker.
        return (delta < 0.025) & (maximum >= 0.30) & (maximum <= 0.50)

    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-6
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    sel = nonzero & (maximum == r)
    hue[sel] = ((g[sel] - b[sel]) / delta[sel]) % 6
    sel = nonzero & (maximum == g)
    hue[sel] = (b[sel] - r[sel]) / delta[sel] + 2
    sel = nonzero & (maximum == b)
    hue[sel] = (r[sel] - g[sel]) / delta[sel] + 4
    hue /= 6
    target = _hue(COLOR_RGB[color])
    distance = np.minimum(abs(hue - target), 1.0 - abs(hue - target))
    return (distance < 0.055) & (saturation > 0.28) & (maximum > 0.12)


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Connected components as (area, left, top, right, bottom), without scipy."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    for row, col in zip(*np.nonzero(mask)):
        if seen[row, col]:
            continue
        stack = [(int(row), int(col))]
        seen[row, col] = True
        area = 0
        left = right = int(col)
        top = bottom = int(row)
        while stack:
            y, x = stack.pop()
            area += 1
            left, right = min(left, x), max(right, x)
            top, bottom = min(top, y), max(bottom, y)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        components.append((area, left, top, right, bottom))
    return components


def _detect_bbox(image: np.ndarray, color: str, block_type: str) -> tuple[float, float] | None:
    """Return (horizontal centre px, visible width px) for one unique color."""
    if color == "white":
        # Neutral floor/shadow pixels can join adjacent block bodies into one
        # component. Rank black handles by the amount of neutral block face in a
        # local window instead of assigning every handle inside that component.
        neutral_mask = _color_mask(image, color)
        dark = image.max(axis=2) < 32
        handles = []
        for area, left, top, right, bottom in _components(dark):
            width, height = right - left + 1, bottom - top + 1
            if area >= 60 and 15 <= width <= 45 and 5 <= height <= 18:
                handles.append((area, left, top, right, bottom))
        matches = []
        half_width = 34 if block_type == "cube" else 65
        for handle_area, left, top, right, bottom in handles:
            centre = (left + right) / 2.0
            x0 = max(0, int(round(centre)) - half_width)
            x1 = min(image.shape[1], int(round(centre)) + half_width + 1)
            y0 = max(0, top - 15)
            y1 = min(image.shape[0], bottom + 29)
            support = int(neutral_mask[y0:y1, x0:x1].sum())
            window_area = max(1, (y1 - y0) * (x1 - x0))
            matches.append((support / window_area, handle_area, centre))
        if matches:
            support_fraction, _, centre = max(matches)
            if support_fraction >= 0.20:
                # The body width is intentionally omitted: a neighboring neutral
                # floor patch can contaminate it, while colored blocks provide scale.
                return centre, float("nan")

    candidates = []
    for area, left, top, right, bottom in _components(_color_mask(image, color)):
        width, height = right - left + 1, bottom - top + 1
        fill = area / (width * height)
        expected_ratio = BLOCK_WIDTH_M[block_type] / 0.10
        if area >= 80 and 20 <= width <= 140 and 15 <= height <= 80 and fill >= 0.25:
            shape_penalty = abs(width / height - expected_ratio)
            center_penalty = abs((left + right) / 2 - image.shape[1] / 2) / image.shape[1]
            candidates.append((area - 80 * shape_penalty - 100 * center_penalty, left, right))
    if not candidates:
        return None
    _, left, right = max(candidates)
    return (left + right) / 2.0, float(right - left + 1)


def _sample_text(perceive_cfg: dict) -> str:
    mode = perceive_cfg.get("cache_mode", "fresh_per_seed")
    base = perceive_cfg.get("text", "")
    if mode == "fresh_per_seed":
        # The stage contract does not receive the episode seed. A nonce makes every
        # episode invocation a fresh draw, which is the sampling assumption in H-013.
        return f"{base}\nIndependent-sample nonce: {uuid.uuid4().hex}"
    if mode == "reuse":
        return base
    raise ValueError(f"unsupported perceive.cache_mode {mode!r}")


@stage("perceive", "vlm_measure")
def perceive_vlm_measure(scene, cfg):
    """Infer structure and measure y using target-picture pixels only.

    Exact measurement inputs from ``scene``:
      * ``scene["image_multiview"]``: target-picture pixels sent to the VLM;
      * ``scene["image_views"]["robot side"]``: side-elevation target pixels used
        for color masks, bounding-box centres, pixel scale, and bottom-layer origin.

    No target/spec/position field, scene JSON position, or simulator state is read.
    Known task constants are block widths (cube 0.12 m, brick 0.24 m), x=0,
    yaw=0, and z=0.05+0.10*layer. Missing color masks fall back to VLM rough y.
    """
    vlm_picture, measurement_picture = _picture_inputs(scene)
    perceive_cfg = cfg["perceive"]
    sample_text = _sample_text(perceive_cfg)
    result = call_model(
        perceive_cfg["model"],
        perceive_cfg["prompt"],
        text=sample_text,
        image=str(vlm_picture),
        use_cache=bool(perceive_cfg.get("cached", True)),
    )
    vlm_blocks = result["output"]["blocks"]
    image = np.asarray(Image.open(measurement_picture).convert("RGB"))

    detections = {
        block["id"]: _detect_bbox(image, block["color"], block["type"])
        for block in vlm_blocks
    }
    scales = [
        detection[1] / BLOCK_WIDTH_M[block["type"]]
        for block in vlm_blocks
        if (detection := detections[block["id"]]) is not None
        and np.isfinite(detection[1])
    ]
    pixels_per_m = float(np.median(scales)) if scales else None
    bottom_centres = [
        detections[block["id"]][0]
        for block in vlm_blocks
        if block["layer"] == 0 and detections[block["id"]] is not None
    ]
    bottom_reference = (
        (min(bottom_centres) + max(bottom_centres)) / 2.0
        if bottom_centres
        else None
    )

    blocks = []
    for index, parsed in enumerate(vlm_blocks):
        block = dict(parsed)
        block.setdefault("id", f"p{index}")
        detection = detections.get(block["id"])
        fallback = detection is None or pixels_per_m is None or bottom_reference is None
        if fallback:
            lateral_y = float(block["rough_y_cm"]) / 100.0
        else:
            # In the robot-side view, increasing structure y points left in the image.
            lateral_y = -(detection[0] - bottom_reference) / pixels_per_m
        block["pos"] = [0.0, float(lateral_y), 0.05 + 0.10 * int(block["layer"])]
        block["yaw"] = 0.0
        block["on"] = list(block["supported_by"])
        block["lateral_measurement_fallback"] = fallback
        block["lateral_measurement_source"] = "vlm_rough_y" if fallback else "target_pixels"
        blocks.append(block)

    return blocks, {
        "model_key": result["key"],
        "cached": bool(result["cached"]),
        "cache_mode": perceive_cfg.get("cache_mode", "fresh_per_seed"),
        "prompt_sha256": result["prompt_sha256"],
        "image_sha256": result["image_sha256"],
        "response_sha256": result["response_sha256"],
        "vlm_output_sha256": result["response_sha256"],
        "measurement_view": "robot side",
        "pixels_per_m": pixels_per_m,
        "fallback_blocks": sum(b["lateral_measurement_fallback"] for b in blocks),
    }
