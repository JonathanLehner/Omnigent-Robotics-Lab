import numpy as np

from lab import pipeline
from lab.stages.walk_head import walk_end_telemetry


def test_v2_walk_head_is_opt_in_and_matches_h023_config():
    cfg = pipeline.load_method("v2_walk_head")

    assert cfg["max_workers"] == 6
    assert cfg["stages"]["build"] == "sumo_walk_head_then_scripted_weld"
    assert cfg["sumo"]["task"] == "spot_walk_approach_head"
    assert "settle_phase" not in cfg["sumo"]
    assert cfg["idealizations"] == [
        "oracle_structure_spec",
        "oracle_build_order",
        "weld_grasp",
        "scripted_arm",
        "no_arm_block_collision",
        "sumo_to_assembly_state_bridge",
        "fixed_nominal_walk_start",
    ]
    assert cfg["sumo"]["task_config"] == {
        "w_position": 160.0,
        "w_heading": 160.0,
        "w_controls": 0.2,
        "w_terminal_xy": 400.0,
        "terminal_fraction": 0.25,
        "w_base_speed": 10.0,
        "base_speed_cap": 0.0,
        "base_speed_radius": 0.10,
        "w_terminal_heading": 400.0,
        "w_terminal_speed": 50.0,
    }


def test_walk_end_telemetry_logs_all_h023_eligibility_values():
    yaw = np.radians(-11.0) / 2.0
    episode = {
        "terminal_base_pose": [
            -0.13,
            0.01,
            0.3,
            np.cos(yaw),
            0.0,
            0.0,
            np.sin(yaw),
        ],
        "nominal_base_goal_pose": [
            -0.15,
            0.0,
            0.3,
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        "terminal_base_speed_m_s": 0.021,
    }

    telemetry = walk_end_telemetry(episode)

    assert np.isclose(telemetry["xy_error_m"], np.sqrt(0.02**2 + 0.01**2))
    assert np.isclose(telemetry["xy_error_cm"], np.sqrt(5.0))
    assert np.isclose(telemetry["heading_error_deg"], -11.0)
    assert telemetry["yaw_error_deg"] == telemetry["heading_error_deg"]
    assert telemetry["planar_base_speed_m_s"] == 0.021
