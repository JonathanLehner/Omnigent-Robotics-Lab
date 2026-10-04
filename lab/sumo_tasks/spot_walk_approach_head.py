"""Walking approach with terminal XY, heading, and speed objectives."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from judo.tasks import register_task

from lab.sumo_tasks.spot_walk_approach_xy import (
    SpotWalkApproachXY,
    SpotWalkApproachXYConfig,
)
from sumo.tasks import _SPOT_REGISTRATION_KWARGS


@dataclass
class SpotWalkApproachHeadConfig(SpotWalkApproachXYConfig):
    """H-023 terminal-window cost weights."""

    w_terminal_heading: float = 0.0
    w_terminal_speed: float = 0.0


class SpotWalkApproachHead(SpotWalkApproachXY):
    """Add linear heading and planar-speed costs in the terminal window."""

    name = "spot_walk_approach_head"
    config_t: type[SpotWalkApproachHeadConfig] = SpotWalkApproachHeadConfig
    config: SpotWalkApproachHeadConfig

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        total = super().reward(states, sensors, controls, system_metadata)
        qpos = states[..., : self.model.nq]
        qvel = states[..., self.model.nq :]
        terminal_steps = max(
            1, int(np.ceil(states.shape[-2] * self.config.terminal_fraction))
        )
        terminal = slice(-terminal_steps, None)

        quat = qpos[
            ..., self.body_pose_idx + 3 : self.body_pose_idx + 7
        ]
        yaw_error = np.arctan2(
            2.0 * (quat[..., 0] * quat[..., 3] + quat[..., 1] * quat[..., 2]),
            1.0 - 2.0 * (np.square(quat[..., 2]) + np.square(quat[..., 3])),
        )
        terminal_heading_cost = self.config.w_terminal_heading * np.abs(
            yaw_error[..., terminal]
        ).mean(axis=-1)

        base_speed = np.linalg.norm(
            qvel[..., self.body_vel_idx : self.body_vel_idx + 2], axis=-1
        )
        terminal_speed_cost = self.config.w_terminal_speed * base_speed[
            ..., terminal
        ].mean(axis=-1)

        total = total - terminal_heading_cost - terminal_speed_cost
        assert total.shape == (states.shape[0],)
        return total


register_task(
    SpotWalkApproachHead.name,
    SpotWalkApproachHead,
    SpotWalkApproachHeadConfig,
    **_SPOT_REGISTRATION_KWARGS,
)
