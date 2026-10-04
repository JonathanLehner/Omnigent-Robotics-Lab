import copy
import math

import numpy as np
from PIL import Image

from lab import sim
from lab.stages import vlm_measure


class GuardedScene(dict):
    """Fail immediately if the stage asks for anything but target-picture paths."""

    def __getitem__(self, key):
        assert key in {"image_multiview", "image_views"}, f"forbidden scene access: {key}"
        return super().__getitem__(key)


def test_vlm_measure_is_invariant_to_deleted_or_poisoned_oracle_fields(tmp_path, monkeypatch):
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[60:100, 55:105] = [83, 15, 15]
    Image.fromarray(image).save(tmp_path / "target.png")
    parsed = {
        "blocks": [{
            "id": "seen-red",
            "color": "red",
            "type": "cube",
            "layer": 0,
            "supported_by": [],
            "rough_y_cm": 19.0,
        }]
    }

    def fake_call(*args, **kwargs):
        return {
            "output": copy.deepcopy(parsed),
            "key": "mock",
            "cached": False,
            "prompt_sha256": "1" * 64,
            "image_sha256": "2" * 64,
            "response_sha256": "3" * 64,
        }

    monkeypatch.setattr(vlm_measure, "ROOT", tmp_path)
    monkeypatch.setattr(vlm_measure, "call_model", fake_call)
    cfg = {
        "perceive": {
            "model": "codex",
            "prompt": "prompts/perceive/v2.md",
            "cache_mode": "reuse",
        }
    }
    picture_fields = {
        "image_multiview": "target.png",
        "image_views": {"robot side": "target.png"},
    }
    deleted = GuardedScene(picture_fields)
    poisoned = GuardedScene({
        **picture_fields,
        "target": {"blocks": [{"pos": [math.nan] * 3}]},
        "spec": [{"position": [math.nan] * 3}],
        "position": [math.nan] * 3,
    })

    clean_output = vlm_measure.perceive_vlm_measure(deleted, cfg)
    monkeypatch.setattr(sim, "SITE", np.full(3, math.nan))
    poisoned_output = vlm_measure.perceive_vlm_measure(poisoned, cfg)
    assert clean_output == poisoned_output
    assert clean_output[0][0]["pos"] == [0.0, 0.0, 0.05]
    assert clean_output[0][0]["lateral_measurement_fallback"] is False
    assert clean_output[1]["cached"] is False
    assert len(clean_output[1]["prompt_sha256"]) == 64
    assert len(clean_output[1]["image_sha256"]) == 64
    assert len(clean_output[1]["response_sha256"]) == 64
    assert clean_output[1]["vlm_output_sha256"] == poisoned_output[1]["vlm_output_sha256"]
