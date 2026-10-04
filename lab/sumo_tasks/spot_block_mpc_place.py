"""Relic-in-the-loop MPC placement of a free block with an agent-written cost.

The public Sumo box model is used as the handled-block dynamics proxy.  The
cost is deliberately independent of ``spot_box_push.reward`` and the terminal
predicate is independent of its stale ``goal_pos`` config reference.
"""

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from judo.config import set_config_overrides
from judo.tasks import register_task
from mujoco import MjData, MjModel

from sumo.controller import ControllerConfig
from sumo.tasks import _SPOT_REGISTRATION_KWARGS
from sumo.tasks.spot.spot_box_push import SpotBoxPush, SpotBoxPushConfig
from sumo.tasks.spot.spot_constants import LEGS_STANDING_POS, STANDING_HEIGHT


@dataclass
class SpotBlockMPCPlaceConfig(SpotBoxPushConfig):
    """Weights and terminal tolerances for object placement."""

    start_x: float = -0.45
    start_y: float = 0.0
    start_z: float = 0.05
    start_yaw_deg: float = 0.0
    approach_distance_m: float = 0.8
    # Direct run_sumo_task smoke defaults are T0-dev-01. Pipeline runs replace
    # these with the current scene's block geometry, mass, and target.
    goal_x: float = 0.85
    goal_y: float = 0.0
    goal_z: float = 0.05
    goal_yaw_deg: float = 0.0
    object_half_height: float = 0.05
    object_size: tuple[float, float, float] = (0.12, 0.24, 0.10)
    object_mass: float = 0.8
    w_place_position: float = 220.0
    w_place_orientation: float = 80.0
    w_upright: float = 280.0
    w_collision: float = 250.0
    w_object_settle: float = 18.0
    w_object_angular_velocity: float = 45.0
    w_gripper_object: float = 6.0
    w_gripper_height: float = 80.0
    w_base_stability: float = 100.0
    w_action_rate: float = 0.25
    # Match the frozen assembly metric so Sumo cannot stop on a pose that the
    # lab world would reject after handoff and settling.
    place_position_tolerance: float = 0.03
    place_angle_tolerance_deg: float = 10.0
    settle_speed_tolerance: float = 0.25
    min_base_clearance: float = 0.32
    # SpotBoxPushConfig does not inherit the locomotion task's threshold even
    # though several upstream tasks assume it does.
    spot_fallen_threshold: float = 0.30


class SpotBlockMPCPlace(SpotBoxPush):
    """Relic-in-the-loop MPC task with an agent-written placement cost."""

    name = "spot_block_mpc_place"
    config_t: type[SpotBlockMPCPlaceConfig] = SpotBlockMPCPlaceConfig
    config: SpotBlockMPCPlaceConfig

    def __init__(self, config: SpotBlockMPCPlaceConfig | None = None) -> None:
        super().__init__(config=config)
        self.configure_object_proxy(
            np.asarray(self.config.object_size), self.config.object_mass
        )
        self.object_vel_idx = self.model.jnt_dofadr[
            self.model.joint("box_joint").id
        ]

    def configure_object_proxy(
        self, size: np.ndarray, mass: float
    ) -> None:
        """Set the handled proxy's full dimensions, mass, and box inertia."""
        size = np.asarray(size, dtype=float)
        half_size = size / 2.0
        for geom_name in ("box_collision", "box_visual"):
            self.sim_model.geom(geom_name).size[:] = half_size
        body_id = self.sim_model.body("box_body").id
        self.sim_model.body_mass[body_id] = mass
        self.sim_model.body_inertia[body_id] = (
            mass
            / 3.0
            * np.array(
                [
                    half_size[1] ** 2 + half_size[2] ** 2,
                    half_size[0] ** 2 + half_size[2] ** 2,
                    half_size[0] ** 2 + half_size[1] ** 2,
                ]
            )
        )
        self.config.object_half_height = float(half_size[2])
        mujoco.mj_setConst(self.sim_model, self.data)

    @property
    def reset_pose(self) -> np.ndarray:
        """Deterministic T0 placement setup: Spot, one block, one nearby goal."""
        push = np.array(
            [
                self.config.goal_x - self.config.start_x,
                self.config.goal_y - self.config.start_y,
            ]
        )
        push_norm = np.linalg.norm(push)
        push_dir = push / push_norm if push_norm > 1e-9 else np.array([1.0, 0.0])
        robot_xy = (
            np.array([self.config.start_x, self.config.start_y])
            - self.config.approach_distance_m * push_dir
        )
        robot_half_yaw = np.arctan2(push_dir[1], push_dir[0]) / 2.0
        robot_pose = np.array(
            [
                robot_xy[0],
                robot_xy[1],
                STANDING_HEIGHT,
                np.cos(robot_half_yaw),
                0.0,
                0.0,
                np.sin(robot_half_yaw),
            ]
        )
        box_half_yaw = np.radians(self.config.start_yaw_deg) / 2.0
        box_pose = np.array(
            [
                self.config.start_x,
                self.config.start_y,
                self.config.start_z,
                np.cos(box_half_yaw),
                0.0,
                0.0,
                np.sin(box_half_yaw),
            ]
        )
        return np.array(
            [
                *robot_pose,
                *LEGS_STANDING_POS,
                *self.reset_arm_pos,
                *box_pose,
            ]
        )

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Score pose error, settling, reachability, stability, and effort."""
        del system_metadata
        batch_size = states.shape[0]
        qpos = states[..., : self.model.nq]
        qvel = states[..., self.model.nq :]

        placed = qpos[..., self.object_pose_idx : self.object_pose_idx + 3]
        goal = np.array(
            [self.config.goal_x, self.config.goal_y, self.config.goal_z]
        )
        pose_cost = self.config.w_place_position * np.linalg.norm(
            placed - goal, axis=-1
        ).mean(axis=-1)

        # Quaternion error to the proxy-frame target (sign invariant).
        placed_quat = qpos[
            ..., self.object_pose_idx + 3 : self.object_pose_idx + 7
        ]
        half_yaw = np.radians(self.config.goal_yaw_deg) / 2.0
        goal_quat = np.array(
            [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]
        )
        orientation_cost = self.config.w_place_orientation * (
            1.0 - np.square(np.sum(placed_quat * goal_quat, axis=-1))
        ).mean(axis=-1)

        # Preserve the box's vertical axis throughout the push.  The full
        # quaternion cost above was too cheap relative to translation: a 90
        # degree tip cost only half of w_place_orientation, so CEM could profit
        # by contacting the upper face and rolling the box toward the goal.
        # R_zz is the dot product between the block and world z axes.
        quat_norm_sq = np.maximum(
            np.square(placed_quat).sum(axis=-1), 1e-12
        )
        block_z_dot_world_z = 1.0 - (
            2.0
            * (
                np.square(placed_quat[..., 1])
                + np.square(placed_quat[..., 2])
            )
            / quat_norm_sq
        )
        upright_cost = self.config.w_upright * (
            1.0 - np.clip(block_z_dot_world_z, -1.0, 1.0)
        ).mean(axis=-1)

        placed_velocity = qvel[
            ..., self.object_vel_idx : self.object_vel_idx + 6
        ]
        settle_cost = self.config.w_object_settle * np.linalg.norm(
            placed_velocity, axis=-1
        ).mean(axis=-1)
        angular_velocity_cost = (
            self.config.w_object_angular_velocity
            * np.linalg.norm(placed_velocity[..., 3:], axis=-1).mean(axis=-1)
        )

        gripper = sensors[..., self.gripper_pos_idx : self.gripper_pos_idx + 3]
        gripper_cost = self.config.w_gripper_object * np.linalg.norm(
            gripper - placed, axis=-1
        ).mean(axis=-1)
        # A horizontal push through the proxy's centre avoids the overturning
        # moment observed when the finger trace stayed 0.343 m above the CoM.
        gripper_height_cost = self.config.w_gripper_height * np.abs(
            gripper[..., 2] - placed[..., 2]
        ).mean(axis=-1)

        base_height = qpos[..., self.body_pose_idx + 2]
        stability_cost = self.config.w_base_stability * np.square(
            base_height - STANDING_HEIGHT
        ).mean(axis=-1)
        base_xy = qpos[..., self.body_pose_idx : self.body_pose_idx + 2]
        clearance = np.linalg.norm(base_xy - placed[..., :2], axis=-1)
        collision_cost = self.config.w_collision * np.square(
            np.maximum(0.0, self.config.min_base_clearance - clearance)
        ).mean(axis=-1)
        collision_cost += self.config.w_collision * np.square(
            np.maximum(0.0, self.config.object_half_height - placed[..., 2])
        ).mean(axis=-1)
        effort_cost = self.config.w_action_rate * np.square(controls).sum(
            axis=-1
        ).mean(axis=-1)

        total = -(
            pose_cost
            + orientation_cost
            + upright_cost
            + collision_cost
            + settle_cost
            + angular_velocity_cost
            + gripper_cost
            + gripper_height_cost
            + stability_cost
            + effort_cost
        )
        assert total.shape == (batch_size,)
        return total

    def success(
        self,
        model: MjModel,
        data: MjData,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Read current MuJoCo state; do not use the stale box-push config."""
        del model, metadata
        placed = data.qpos[self.object_pose_idx : self.object_pose_idx + 3]
        goal = np.array(
            [self.config.goal_x, self.config.goal_y, self.config.goal_z]
        )
        pose_ok = (
            np.linalg.norm(placed - goal)
            <= self.config.place_position_tolerance
        )
        quat = data.qpos[
            self.object_pose_idx + 3 : self.object_pose_idx + 7
        ]
        half_yaw = np.radians(self.config.goal_yaw_deg) / 2.0
        goal_quat = np.array(
            [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]
        )
        angle_deg = np.degrees(
            2.0
            * np.arccos(
                np.clip(abs(np.dot(quat, goal_quat)), 0.0, 1.0)
            )
        )
        speed = np.linalg.norm(
            data.qvel[self.object_vel_idx : self.object_vel_idx + 6]
        )
        standing = (
            data.qpos[self.body_pose_idx + 2]
            > self.config.spot_fallen_threshold
        )
        return bool(
            pose_ok
            and angle_deg <= self.config.place_angle_tolerance_deg
            and speed <= self.config.settle_speed_tolerance
            and standing
        )

    def failure(
        self,
        model: MjModel,
        data: MjData,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Terminate only when Spot has fallen."""
        del model, metadata
        return bool(
            data.qpos[self.body_pose_idx + 2] <= self.config.spot_fallen_threshold
        )


register_task(
    SpotBlockMPCPlace.name,
    SpotBlockMPCPlace,
    SpotBlockMPCPlaceConfig,
    **_SPOT_REGISTRATION_KWARGS,
)
set_config_overrides(
    SpotBlockMPCPlace.name,
    ControllerConfig,
    {"horizon": 2.0},
)
