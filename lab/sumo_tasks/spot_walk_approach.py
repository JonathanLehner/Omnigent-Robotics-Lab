"""Sumo/Relic whole-body walking approach task for rung 2."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from judo.tasks import register_task
from mujoco import MjData, MjModel

from sumo.tasks import _SPOT_REGISTRATION_KWARGS
from sumo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from sumo.tasks.spot.spot_constants import LEGS_STANDING_POS_RL, STANDING_HEIGHT


@dataclass
class SpotWalkApproachConfig(SpotBaseConfig):
    """Cost weights and the build-site approach tolerance."""

    # The build site is the origin in this task frame.  Reset 1.5 m away and
    # approach a 0.15 m standoff, leaving a 1.35 m commanded walk.
    start_x: float = -1.5
    start_y: float = 0.0
    start_yaw: float = 0.0
    goal_x: float = -0.15
    goal_y: float = 0.0
    goal_tolerance: float = 0.25
    w_position: float = 160.0
    # Match the position weight: the old 8.0 cost let CEM trade a 90-degree
    # arrival yaw for only ~4 cost units while still landing on the goal XY.
    # The assembly bridge preserves that real yaw, which can put the block
    # outside the arm workspace even though the walk itself reports success.
    w_heading: float = 160.0
    w_controls: float = 0.2
    fall_penalty: float = 10_000.0


class SpotWalkApproach(SpotBase[SpotWalkApproachConfig]):
    """Walk from a displaced reset pose to a build-site standoff pose."""

    name = "spot_walk_approach"
    config_t: type[SpotWalkApproachConfig] = SpotWalkApproachConfig
    config: SpotWalkApproachConfig

    def __init__(self, config: SpotWalkApproachConfig | None = None) -> None:
        super().__init__(use_arm=False, config=config)

    @property
    def reset_pose(self) -> np.ndarray:
        half_yaw = self.config.start_yaw / 2.0
        return np.array(
            [
                self.config.start_x,
                self.config.start_y,
                STANDING_HEIGHT,
                np.cos(half_yaw),
                0.0,
                0.0,
                np.sin(half_yaw),
                *LEGS_STANDING_POS_RL,
                *self.reset_arm_pos,
            ]
        )

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        del sensors, system_metadata
        qpos = states[..., : self.model.nq]
        base_xy = qpos[..., self.body_pose_idx : self.body_pose_idx + 2]
        goal = np.array([self.config.goal_x, self.config.goal_y])
        distance_cost = self.config.w_position * np.linalg.norm(
            base_xy - goal, axis=-1
        ).mean(axis=-1)
        # Penalize yaw through the z component of the base quaternion.
        heading_cost = self.config.w_heading * np.square(
            qpos[..., self.body_pose_idx + 6]
        ).mean(axis=-1)
        effort_cost = self.config.w_controls * np.square(controls).sum(
            axis=-1
        ).mean(axis=-1)
        fallen = (
            qpos[..., self.body_pose_idx + 2] <= self.config.spot_fallen_threshold
        ).any(axis=-1)
        total = -distance_cost - heading_cost - effort_cost
        total -= self.config.fall_penalty * fallen
        assert total.shape == (states.shape[0],)
        return total

    def success(
        self,
        model: MjModel,
        data: MjData,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        del model, metadata
        base = data.qpos[self.body_pose_idx : self.body_pose_idx + 3]
        goal = np.array([self.config.goal_x, self.config.goal_y])
        return bool(
            np.linalg.norm(base[:2] - goal) <= self.config.goal_tolerance
            and base[2] > self.config.spot_fallen_threshold
        )

    def failure(
        self,
        model: MjModel,
        data: MjData,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        del model, metadata
        return bool(
            data.qpos[self.body_pose_idx + 2] <= self.config.spot_fallen_threshold
        )


register_task(
    SpotWalkApproach.name,
    SpotWalkApproach,
    SpotWalkApproachConfig,
    **_SPOT_REGISTRATION_KWARGS,
)
