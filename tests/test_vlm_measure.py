import copy
import hashlib
import inspect
import json
import math

import numpy as np
from PIL import Image

from lab import scenes, sim
from lab.stages import vlm_measure
from lab.stages import vlm_measure_anchor
from lab.stages import vlm_measure_telemetry
from lab.stages import vlm_anchor_telemetry


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


def test_white_detector_uses_neutral_face_handle_on_bridge_supports():
    image = np.zeros((100, 180, 3), dtype=np.uint8)
    image[45:85, 20:80] = [54, 25, 74]
    image[45:85, 100:160] = [90, 90, 90]
    image[55:65, 35:65] = 10
    image[55:65, 115:145] = 10

    center, width = vlm_measure._detect_bbox(image, "white", "cube")

    assert center == 129.5
    assert math.isnan(width)


def test_position_error_telemetry_is_perceived_minus_true():
    perceived = [{
        "id": "p0",
        "color": "white",
        "type": "cube",
        "pos": [0.0, 0.07, 0.05],
    }]
    scene = {"target": {"blocks": [{
        "id": "b0",
        "color": "white",
        "type": "cube",
        "pos": [0.0, 0.06, 0.05],
    }]}}

    telemetry = vlm_measure_telemetry._position_errors(perceived, scene)

    assert np.allclose(
        telemetry[0]["error_perceived_minus_true_xyz_m"],
        [0.0, 0.01, 0.0],
    )


def test_vlm_measure_anchored_reads_only_target_picture(tmp_path, monkeypatch):
    image = np.zeros((360, 480, 3), dtype=np.uint8)
    image[175:230, 205:275] = [83, 15, 15]
    image[198:208, 225:255] = 10
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

    monkeypatch.setattr(vlm_measure_anchor, "ROOT", tmp_path)
    monkeypatch.setattr(vlm_measure_anchor, "call_model", fake_call)
    cfg = {
        "perceive": {
            "model": "codex",
            "prompt": "prompts/perceive/v2.md",
            "cache_mode": "reuse",
            "cached": False,
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
        "start": [{"pos": [math.nan] * 3}],
        "simulator_state": math.nan,
    })

    clean_output = vlm_measure_anchor.perceive_vlm_measure_anchored(deleted, cfg)
    poisoned_output = vlm_measure_anchor.perceive_vlm_measure_anchored(poisoned, cfg)

    assert clean_output == poisoned_output
    assert clean_output[0][0]["lateral_measurement_fallback"] is False
    assert clean_output[0][0]["lateral_measurement_source"] == (
        "target_pixels_camera_anchor"
    )
    assert clean_output[1]["cached"] is False


def test_anchor_camera_constants_match_renderer():
    render_signature = inspect.signature(sim.render_structure_views)
    model = sim.build_model([], robot=False)

    assert vlm_measure_anchor.CAMERA_VIEW == tuple(
        float(value) for value in sim.STRUCTURE_VIEWS["robot side"]
    )
    assert vlm_measure_anchor.IMAGE_WIDTH_PX == render_signature.parameters["w"].default
    assert vlm_measure_anchor.IMAGE_HEIGHT_PX == render_signature.parameters["h"].default
    assert vlm_measure_anchor.CAMERA_FOVY_DEG == model.vis.global_.fovy


def test_fresh_draw_prompt_bytes_are_pinned():
    cfg = {"cache_mode": "fresh_per_seed", "text": "fixed"}
    assert vlm_measure._sample_text(cfg) == "fixed"
    assert vlm_measure._sample_text(cfg) == "fixed"


def test_anchor_telemetry_reports_missing_spurious_e_and_p():
    spec = [
        {
            "id": "p0",
            "color": "red",
            "type": "cube",
            "pos": [0.0, 0.06, 0.05],
        },
        {
            "id": "p1",
            "color": "blue",
            "type": "cube",
            "pos": [0.0, -0.02, 0.05],
        },
    ]
    scene = {
        "target": {
            "blocks": [
                {
                    "id": "b0",
                    "color": "red",
                    "type": "cube",
                    "pos": [0.0, 0.05, 0.05],
                },
                {
                    "id": "b1",
                    "color": "white",
                    "type": "cube",
                    "pos": [0.0, -0.05, 0.05],
                },
            ]
        }
    }
    info = {
        "ppm_blocks": 420.0,
        "block_measurements": [{
            "perceived_id": "p0",
            "u_px": 215.0,
            "measurement_point": "handle_front_center",
            "d_m": 1.0,
            "y_anchored": 0.06,
            "y_recentred": 0.0,
            "ppm_camera": 434.0,
            "ppm_blocks": 420.0,
        }],
    }
    pairs = {"p0": "b0"}

    vlm_anchor_telemetry.attach_position_errors(info, spec, scene, pairs)
    per_block = [
        {
            "id": "b0",
            "pos_err": 0.011,
            "ang_err": 0.0,
            "final_pose": {
                "position_xyz": [0.85, 0.071, 0.05],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
        },
        {
            "id": "b1",
            "pos_err": 0.20,
            "ang_err": 0.0,
            "final_pose": {
                "position_xyz": [0.85, 0.15, 0.05],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
        },
    ]
    vlm_anchor_telemetry.attach_placement_residuals(info, per_block, spec, pairs)

    matched, missing = info["block_position_errors"]
    assert matched["e_status"] == "matched"
    assert math.isclose(matched["e"], 1.0)
    assert math.isclose(matched["p"], 1.1)
    assert missing["e_status"] == "missing"
    assert missing["e"] is None
    assert missing["p"] is None
    assert info["n_missing"] == 1
    assert info["n_spurious"] == 1
    assert info["spurious_perceived_blocks"][0]["perceived_id"] == "p1"


def test_scene_hashes_are_from_bytes_loaded_once(tmp_path, monkeypatch):
    scene_dir = tmp_path / "scenes" / "dev"
    scene_dir.mkdir(parents=True)
    pictures = {
        "scene.png": b"main",
        "scene_multiview.png": b"multiview",
        "scene_robot_side.png": b"robot-side",
    }
    for name, content in pictures.items():
        (scene_dir / name).write_bytes(content)
    payload = {
        "id": "T0-dev-99",
        "tier": "T0",
        "split": "dev",
        "target": {"blocks": []},
        "start": [],
        "image": "scenes/dev/scene.png",
        "image_multiview": "scenes/dev/scene_multiview.png",
        "image_views": {"robot side": "scenes/dev/scene_robot_side.png"},
    }
    spec_bytes = json.dumps(payload).encode()
    (scene_dir / "T0-dev-99.json").write_bytes(spec_bytes)
    monkeypatch.setattr(scenes, "ROOT", tmp_path)
    monkeypatch.setattr(scenes, "SCENES", tmp_path / "scenes")

    loaded = scenes.load_scene("T0-dev-99")
    (scene_dir / "scene.png").write_bytes(b"changed-after-load")

    hashes = loaded["_loaded_scene_files_sha256"]
    assert hashes["scenes/dev/T0-dev-99.json"] == hashlib.sha256(spec_bytes).hexdigest()
    assert hashes["scenes/dev/scene.png"] == hashlib.sha256(b"main").hexdigest()
    assert len(hashes) == 4


def test_localization_failure_classifier_uses_e_and_p_split():
    info = {
        "block_position_errors": [{
            "true_id": "b0",
            "e_status": "matched",
            "e": 2.1,
            "p": 0.4,
        }]
    }
    failed = [{"id": "b0", "pos_err": 0.04, "ang_err": 0.0}]

    assert vlm_anchor_telemetry.has_localization_failure(info, failed)

    info["block_position_errors"][0]["p"] = 1.6
    assert not vlm_anchor_telemetry.has_localization_failure(info, failed)
