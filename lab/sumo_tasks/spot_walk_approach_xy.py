"""Walking approach variant with terminal-XY and stopping costs."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from judo.tasks import register_task

from lab.sumo_tasks.spot_walk_approach import (
    SpotWalkApproach,
    SpotWalkApproachConfig,
)
from sumo.tasks import _SPOT_REGISTRATION_KWARGS


@dataclass
class SpotWalkApproachXYConfig(SpotWalkApproachConfig):
    """Optional XY/stopping costs used by the H-016 candidates."""

    w_xy_quad: float = 0.0
    w_terminal_xy: float = 0.0
    terminal_fraction: float = 0.25
    squared_xy: bool = False
    w_base_speed: float = 0.0
    base_speed_cap: float = 0.0
    base_speed_radius: float = 0.0


class SpotWalkApproachXY(SpotWalkApproach):
    """Approach task with an explicit terminal-position objective."""

    name = "spot_walk_approach_xy"
    config_t: type[SpotWalkApproachXYConfig] = SpotWalkApproachXYConfig
    config: SpotWalkApproachXYConfig

    def __init__(self, config: SpotWalkApproachXYConfig | None = None) -> None:
        super().__init__(config=config)
        self.body_vel_idx = self.model.jnt_dofadr[
            self.model.joint("base").id
        ]

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        del sensors, system_metadata
        qpos = states[..., : self.model.nq]
        qvel = states[..., self.model.nq :]
        base_xy = qpos[..., self.body_pose_idx : self.body_pose_idx + 2]
        goal = np.array([self.config.goal_x, self.config.goal_y])
        xy_error_sq = np.square(base_xy - goal).sum(axis=-1)
        if self.config.squared_xy:
            position_cost = self.config.w_position * xy_error_sq.mean(axis=-1)
        else:
            position_cost = self.config.w_position * np.sqrt(
                xy_error_sq
            ).mean(axis=-1)
        xy_quad_cost = self.config.w_xy_quad * xy_error_sq.mean(axis=-1)

        terminal_steps = max(
            1, int(np.ceil(states.shape[-2] * self.config.terminal_fraction))
        )
        terminal_cost = self.config.w_terminal_xy * xy_error_sq[
            ..., -terminal_steps:
        ].mean(axis=-1)

        heading_cost = self.config.w_heading * np.square(
            qpos[..., self.body_pose_idx + 6]
        ).mean(axis=-1)
        effort_cost = self.config.w_controls * np.square(controls).sum(
            axis=-1
        ).mean(axis=-1)

        base_speed = np.linalg.norm(
            qvel[..., self.body_vel_idx : self.body_vel_idx + 2], axis=-1
        )
        speed_excess = np.maximum(
            0.0, base_speed - self.config.base_speed_cap
        )
        if self.config.base_speed_radius > 0.0:
            speed_excess = np.where(
                np.sqrt(xy_error_sq) <= self.config.base_speed_radius,
                speed_excess,
                0.0,
            )
        speed_cost = self.config.w_base_speed * np.square(
            speed_excess
        ).mean(axis=-1)

        fallen = (
            qpos[..., self.body_pose_idx + 2]
            <= self.config.spot_fallen_threshold
        ).any(axis=-1)
        total = (
            -position_cost
            - xy_quad_cost
            - terminal_cost
            - heading_cost
            - effort_cost
            - speed_cost
        )
        total -= self.config.fall_penalty * fallen
        assert total.shape == (states.shape[0],)
        return total


register_task(
    SpotWalkApproachXY.name,
    SpotWalkApproachXY,
    SpotWalkApproachXYConfig,
    **_SPOT_REGISTRATION_KWARGS,
)
