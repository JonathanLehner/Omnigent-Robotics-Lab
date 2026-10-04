"""Regression tests for H-020 composition logging and guards."""

from copy import deepcopy

import pytest

from lab import pipeline, runner
from lab.stages import h020_telemetry

TRUE_BLOCKS = [
    {
        "id": "b0",
        "type": "brick",
        "color": "cyan",
        "pos": [0.0, 0.0, 0.05],
        "yaw": 0.0,
        "on": [],
    },
    {
        "id": "b1",
        "type": "cube",
        "color": "green",
        "pos": [0.0, 0.0, 0.15],
        "yaw": 0.0,
        "on": ["b0"],
    },
]


def _scene():
    return {"target": {"blocks": deepcopy(TRUE_BLOCKS)}}


def test_combo_stage_is_registered():
    cfg = pipeline.load_method("v6_combo")
    assert cfg["stages"]["build"] == "sumo_walk_then_closed_loop_weld_place"
    assert "sumo_walk_then_closed_loop_weld_place" in pipeline.STAGES["build"]


def test_combo_configs_copy_cleared_component_blocks_verbatim():
    branch_a = pipeline.load_method("v6_combo")
    branch_b = pipeline.load_method("v6_combo_t0")
    v2 = pipeline.load_method("v2_walk")
    v47 = pipeline.load_method("v47_vlm_anchor")
    v5 = pipeline.load_method("v5_vlm")

    assert branch_a["perceive"] == v47["perceive"]
    assert branch_a["sumo"] == v2["sumo"]
    assert branch_a["params"] == v47["params"]
    assert branch_b["perceive"] == v5["perceive"]
    assert branch_b["sumo"] == v2["sumo"]
    assert branch_b["params"] == v2["params"]


def test_per_block_pose_errors_and_true_support_logging():
    spec = deepcopy(TRUE_BLOCKS)
    spec[1]["pos"][1] = 0.02
    info = {}
    pairs = {"b0": "b0", "b1": "b1"}

    h020_telemetry.attach_perception_errors(info, spec, _scene(), pairs)
    log = {"perceive": info}
    h020_telemetry.attach_order(log, ["b0", "b1"], spec, _scene(), pairs)

    assert info["block_pose_errors"][1]["e"] == {
        "xyz_cm": [0.0, 2.0, 0.0],
        "yaw_deg": 0.0,
    }
    assert log["plan_order"]["support_valid_perceived"] is True
    assert log["plan_order"]["support_valid_true"] is True

    h020_telemetry.attach_order(log, ["b1", "b0"], spec, _scene(), pairs)
    assert log["plan_order"]["support_valid_true"] is False

    mismatched = deepcopy(spec)
    mismatched[1]["layer"] = 0
    info = {}
    h020_telemetry.attach_perception_errors(
        info, mismatched, _scene(), pairs
    )
    assert info["structure_matches_true"] is False
    log = {"perceive": info}
    h020_telemetry.attach_order(
        log, ["b0", "b1"], mismatched, _scene(), pairs
    )
    assert log["plan_order"]["support_valid_true"] == "n/a_mismatch"


def test_failure_classifier_order_and_walk_absorption():
    base = {
        "success": False,
        "plan_order": {
            "support_valid_true": False,
            "support_valid_perceived": False,
            "planner_error": "order_infeasible: no ready block",
        },
        "walk_end": {"xy_error_cm": 4.0, "yaw_error_deg": 0.0},
        "perceive": {
            "structure_matches_true": False,
            "block_pose_errors": [],
            "spurious_perceived_blocks": [{"perceived_id": "p9"}],
        },
        "blocks": [],
    }
    assert h020_telemetry.classify_failure(base) == "perception_structure"

    matched = deepcopy(base)
    matched["perceive"]["structure_matches_true"] = True
    matched["perceive"]["spurious_perceived_blocks"] = []
    assert h020_telemetry.classify_failure(matched) == "order"

    passing = deepcopy(base)
    passing["success"] = True
    assert h020_telemetry.classify_failure(passing) == "walk_drift_absorbed"


def test_failure_classifier_localization_before_walk():
    row = {
        "true_id": "b0",
        "e_status": "matched",
        "true_type": "cube",
        "perceived_type": "cube",
        "true_layer": 0,
        "perceived_layer": 0,
        "e": {"xyz_cm": [0.0, 2.0, 0.0], "yaw_deg": 0.0},
        "p": {"xyz_cm": [0.0, 1.0, 0.0], "yaw_deg": 0.0},
    }
    log = {
        "success": False,
        "plan_order": {"support_valid_true": True},
        "walk_end": {"xy_error_cm": 4.0, "yaw_error_deg": 0.0},
        "perceive": {
            "block_pose_errors": [row],
            "spurious_perceived_blocks": [],
        },
        "blocks": [{"id": "b0", "pos_err": 0.04, "ang_err": 0.0}],
    }
    assert h020_telemetry.classify_failure(log) == "perception_localization"


def test_scene_hash_coverage_and_experiment_caps():
    scene = {
        "id": "T0-dev-01",
        "split": "dev",
        "image": "scenes/dev/main.png",
        "image_multiview": "scenes/dev/multiview.png",
        "image_views": {
            "main": "scenes/dev/main-view.png",
            "top": "scenes/dev/top.png",
            "robot side": "scenes/dev/robot.png",
            "left side": "scenes/dev/left.png",
        },
    }
    expected = {
        "scenes/dev/T0-dev-01.json",
        scene["image"],
        scene["image_multiview"],
        *scene["image_views"].values(),
    }
    scene["_loaded_scene_files_sha256"] = {path: "abc" for path in expected}
    runner._check_scene_hash_coverage(scene, 7)
    scene["_loaded_scene_files_sha256"].pop("scenes/dev/top.png")
    with pytest.raises(RuntimeError, match="scene_hash_coverage"):
        runner._check_scene_hash_coverage(scene, 7)

    experiment = {"data": {"method": "v6_combo vs v0", "episodes": 192}}
    caps = runner._experiment_arm_caps(experiment)
    assert caps == {"v6_combo": 96, "v0": 96}
    assert all(96 <= runner._method_arm_cap(experiment, method) for method in caps)
    assert all(96 + 1 > runner._method_arm_cap(experiment, method) for method in caps)
    assert runner._method_arm_cap(experiment, "v47_vlm_anchor") == 0
