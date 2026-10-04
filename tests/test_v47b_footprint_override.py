import copy
import math

import numpy as np
from PIL import Image

from lab.h026_offline import (
    majority_vote_types,
    score_structure,
    validate_target_manifest,
)
from lab.stages import v47b_footprint_override


def _image_with_face(width: int, color=(217, 38, 38)) -> np.ndarray:
    image = np.zeros((360, 480, 3), dtype=np.uint8)
    left = 240 - width // 2
    right = left + width
    image[180:230, left:right] = color
    image[198:208, 225:255] = 10
    return image


def _cfg() -> dict:
    return {
        "perceive": {
            "type_override": {
                "nominal_cube_edge_m": 0.12,
                "nominal_brick_length_m": 0.24,
                "cube_band_fraction": 0.25,
                "brick_band_fraction": 0.75,
            }
        }
    }


def _block(block_type: str) -> dict:
    return {
        "id": "red",
        "color": "red",
        "type": block_type,
        "layer": 0,
        "supported_by": [],
        "rough_y_cm": 0,
    }


def test_type_override_fires_only_outside_fixed_middle_band(tmp_path, monkeypatch):
    monkeypatch.setattr(
        v47b_footprint_override,
        "_depth_and_scale",
        lambda _layer, _point: (1.0, 400.0),
    )
    cases = [
        (48, "brick", "cube", True),
        (72, "brick", "brick", False),  # q=1.5: ambiguous, keep VLM type
        (96, "cube", "brick", True),
    ]
    for index, (width, vlm_type, final_type, fired) in enumerate(cases):
        path = tmp_path / f"{index}.png"
        Image.fromarray(_image_with_face(width)).save(path)
        blocks, rows = v47b_footprint_override.override_types_from_picture(
            [_block(vlm_type)], path, _cfg()["perceive"]["type_override"]
        )
        assert blocks[0]["type"] == final_type
        assert rows[0]["vlm_type"] == vlm_type
        assert rows[0]["final_type"] == final_type
        assert rows[0]["override_fired"] is fired
        assert math.isclose(rows[0]["measured_length_m"], width / 400)


def test_every_block_has_complete_type_telemetry_on_real_target():
    from lab import pipeline, scenes

    cfg = pipeline.load_method("v6_combo_v47b")
    scene = scenes.load_scene("T2-dev-11")
    raw = copy.deepcopy(scene["target"]["blocks"])
    for block in raw:
        block["layer"] = round((block["pos"][2] - 0.05) / 0.10)
        block["supported_by"] = list(block["on"])
        block["rough_y_cm"] = block["pos"][1] * 100
        block.pop("pos")
        block.pop("yaw")
    blocks, info = v47b_footprint_override.apply_v47b_to_response(
        raw,
        pipeline.ROOT / scene["image_views"]["robot side"],
        cfg,
    )
    assert len(info["type_measurements"]) == len(blocks)
    assert all(
        {
            "vlm_type",
            "measured_length_m",
            "q",
            "geometric_class",
            "final_type",
            "override_fired",
        }
        <= row.keys()
        for row in info["type_measurements"]
    )
    assert [block["type"] for block in blocks] == [
        block["type"] for block in scene["target"]["blocks"]
    ]


def test_majority_vote_and_structure_score_do_not_match_on_type():
    truth = [
        {
            "id": "b0",
            "color": "cyan",
            "type": "brick",
            "pos": [0, 0, 0.05],
            "yaw": 0,
        }
    ]
    parses = [
        [
            {
                "id": f"p{index}",
                "color": "cyan",
                "type": "brick" if index in (1, 2, 3) else "cube",
                "layer": 0,
                "rough_y_cm": 0,
            }
        ]
        for index in range(5)
    ]
    voted = majority_vote_types(parses)
    assert voted[0]["type"] == "brick"
    assert score_structure(parses[0], truth)["type_errors"] == 1
    assert not score_structure(voted, truth)["structure_mismatch"]


def test_target_manifest_is_exactly_spec_derived():
    assert len(validate_target_manifest()) == 58
