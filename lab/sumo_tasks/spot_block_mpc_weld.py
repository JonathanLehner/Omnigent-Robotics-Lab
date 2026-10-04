"""Relic-in-the-loop weld-carry MPC placement for H-018.

The handled block is welded to Spot's fingertip from reset until the
pre-registered release predicate (or forced-release deadline) fires.  Collision
bits separate every robot geom from the block while retaining robot-ground and
block-ground contacts.
"""

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from judo.config import set_config_overrides
from judo.tasks import register_task
from judo.tasks.spot.spot_box_push import XML_PATH as SPOT_BOX_XML
from mujoco import MjData, MjModel

from sumo.controller import ControllerConfig
from sumo.tasks import _SPOT_REGISTRATION_KWARGS
from sumo.tasks.spot.spot_box_push import SpotBoxPush, SpotBoxPushConfig
from sumo.tasks.spot.spot_constants import LEGS_STANDING_POS


IDEALIZATIONS = (
    "oracle_structure_spec",
    "oracle_build_order",
    "oracle_block_pose",
    "weld_grasp",
    "no_arm_block_collision",
    "fixed_base",
    "sumo_to_assembly_state_bridge",
    "oracle_pick",
    "scripted_release_trigger",
    "sumo_box_proxy_dynamics",
)


def _weld_model_path() -> str:
    """Materialize the public Spot-box model with a fingertip/box weld."""
    source_path = Path(SPOT_BOX_XML).resolve()
    source = source_path.read_text()

    def absolute_include(match: re.Match[str]) -> str:
        include = (source_path.parent / match.group(1)).resolve()
        return f'<include file="{include.as_posix()}" />'

    source = re.sub(
        r'<include\s+file="([^"]+)"\s*/>', absolute_include, source
    )
    source = source.replace(
        '<site name="site_object" pos="0 0 0" size="0.01"/>',
        '<site name="site_object" pos="0 0 0" size="0.01"/>\n'
        '      <site name="box_weld_site" pos="0 0 0.08" size="0.005"/>',
    )
    source = source.replace(
        '<body name="body" pos="0 0 0.52">',
        '<body name="body" pos="0 0 0.46">',
    )
    source = source.replace(
        "</mujoco>",
        '  <equality>\n'
        '    <weld name="gripper_box_weld" site1="site_arm_link_fngr" '
        'site2="box_weld_site" active="true" '
        'solref="-3000 -150" solimp="0.99 0.999 0.001"/>\n'
        '    <weld name="fixed_base_weld" body1="body" active="true" '
        'solref="-2000 -100" solimp="0.95 0.99 0.001"/>\n'
        "  </equality>\n"
        "</mujoco>",
    )
    path = Path(tempfile.gettempdir()) / "spot_block_mpc_weld_h018.xml"
    if not path.exists() or path.read_text() != source:
        path.write_text(source)
    return str(path)


@dataclass
class SpotBlockMPCWeldConfig(SpotBoxPushConfig):
    """Frozen H-018 costs, release thresholds, and task geometry."""

    start_x: float = 0.35
    start_y: float = 0.0
    start_z: float = 0.05
    start_yaw_deg: float = 0.0
    goal_x: float = 0.85
    goal_y: float = 0.0
    goal_z: float = 0.05
    goal_yaw_deg: float = 0.0
    object_half_height: float = 0.05
    object_size: tuple[float, float, float] = (0.12, 0.24, 0.10)
    object_mass: float = 0.8

    w_place_position: float = 220.0
    w_place_quadratic: float = 1000.0
    w_terminal_place: float = 1000.0
    w_place_orientation: float = 80.0
    w_upright: float = 280.0
    w_collision: float = 250.0
    w_object_settle: float = 18.0
    w_object_angular_velocity: float = 45.0
    w_base_stability: float = 100.0
    w_action_rate: float = 0.25
    w_hover: float = 300.0
    w_retreat: float = 50.0
    w_gripper_object: float = 0.0
    w_gripper_height: float = 0.0

    forced_release_s: float = 8.0
    post_release_s: float = 1.0
    release_xy_tolerance_m: float = 0.01
    release_z_lower_m: float = 0.0
    release_z_upper_m: float = 0.015
    release_speed_tolerance_m_s: float = 0.05
    release_yaw_tolerance_deg: float = 5.0
    hover_clearance_m: float = 0.03
    hover_xy_threshold_m: float = 0.02
    spot_fallen_threshold: float = 0.30


class SpotBlockMPCWeld(SpotBoxPush):
    """Carry a welded box with arm MPC while Relic holds a fixed stance."""

    name = "spot_block_mpc_weld"
    config_t: type[SpotBlockMPCWeldConfig] = SpotBlockMPCWeldConfig
    config: SpotBlockMPCWeldConfig

    def __init__(self, config: SpotBlockMPCWeldConfig | None = None) -> None:
        # Bypass the Sumo wrapper's fixed-path convenience initializer while
        # retaining its SpotAssetMixin and public task implementation.
        super(SpotBoxPush, self).__init__(
            model_path=_weld_model_path(), config=config
        )
        self.object_vel_idx = int(
            self.model.jnt_dofadr[self.model.joint("box_joint").id]
        )
        self.weld_eq_id = int(self.model.equality("gripper_box_weld").id)
        self.base_weld_eq_id = int(
            self.model.equality("fixed_base_weld").id
        )
        self.box_body_id = int(self.model.body("box_body").id)
        self.ground_geom_id = int(self.model.geom("ground").id)
        self.finger_site_id = int(self.model.site("site_arm_link_fngr").id)
        self.box_weld_site_id = int(self.model.site("box_weld_site").id)
        self._set_collision_masks()
        self.configure_object_proxy(
            np.asarray(self.config.object_size), self.config.object_mass
        )
        self.release_time_s: float | None = None
        self.forced_release = False
        self.release_reason: str | None = None
        self.release_error: dict[str, float] | None = None
        self.release_pose: list[float] | None = None
        self.release_plus_post_pose: list[float] | None = None
        self.release_gripper_z: float | None = None
        self.release_block_gripper_distance_m: float | None = None
        self.post_release_block_gripper_distance_m: float | None = None
        self.physics_log: list[dict[str, Any]] = []
        self.release_transitions = 0
        self._elapsed_s = 0.0
        self._post_hook_inner = True
        self._initial_base_pose: np.ndarray | None = None
        self._initial_relative_pose: tuple[np.ndarray, np.ndarray] | None = None
        self.reset()

    def _set_collision_masks(self) -> None:
        """Use disjoint robot/block bits and a shared environment bit."""
        box_geom_id = int(self.sim_model.geom("box_collision").id)
        for geom_id in range(self.sim_model.ngeom):
            body_id = int(self.sim_model.geom_bodyid[geom_id])
            if geom_id == box_geom_id:
                self.sim_model.geom_contype[geom_id] = 2
                self.sim_model.geom_conaffinity[geom_id] = 2
            elif body_id == 0:
                # World/support geoms collide with both robot and block.
                self.sim_model.geom_contype[geom_id] = 3
                self.sim_model.geom_conaffinity[geom_id] = 3
            else:
                # All non-box bodies are robot bodies in this proxy model.
                self.sim_model.geom_contype[geom_id] = 1
                self.sim_model.geom_conaffinity[geom_id] = 1
        mujoco.mj_setConst(self.sim_model, self.data)

    def configure_object_proxy(
        self, size: np.ndarray, mass: float
    ) -> None:
        """Set the scene block's full dimensions, mass, and box inertia."""
        size = np.asarray(size, dtype=float)
        half_size = size / 2.0
        for geom_name in ("box_collision", "box_visual"):
            self.sim_model.geom(geom_name).size[:] = half_size
        body_id = int(self.sim_model.body("box_body").id)
        self.sim_model.body_mass[body_id] = float(mass)
        self.sim_model.body_inertia[body_id] = (
            float(mass)
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
        """Reset at the lab fixed-base pose with the box welded in hand."""
        robot_pose = np.array([0.0, 0.0, 0.46, 1.0, 0.0, 0.0, 0.0])
        nominal = np.array(
            [
                *robot_pose,
                *LEGS_STANDING_POS,
                *self.reset_arm_pos,
                self.config.start_x,
                self.config.start_y,
                self.config.start_z,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        scratch = mujoco.MjData(self.sim_model)
        scratch.qpos[:] = nominal
        weld_eq_id = int(
            self.sim_model.equality("gripper_box_weld").id
        )
        finger_site_id = int(
            self.sim_model.site("site_arm_link_fngr").id
        )
        scratch.eq_active[weld_eq_id] = 0
        mujoco.mj_forward(self.sim_model, scratch)
        finger_pos = scratch.site_xpos[finger_site_id].copy()
        finger_quat = np.empty(4)
        mujoco.mju_mat2Quat(
            finger_quat, scratch.site_xmat[finger_site_id]
        )
        weld_offset = np.array([0.0, 0.0, 0.08])
        world_offset = np.empty(3)
        mujoco.mju_rotVecQuat(world_offset, weld_offset, finger_quat)
        object_pose_idx = int(
            self.sim_model.jnt_qposadr[
                self.sim_model.joint("box_joint").id
            ]
        )
        nominal[object_pose_idx : object_pose_idx + 3] = (
            finger_pos - world_offset
        )
        nominal[object_pose_idx + 3 : object_pose_idx + 7] = (
            finger_quat
        )
        return nominal

    def reset(self) -> None:
        if not hasattr(self, "weld_eq_id"):
            super().reset()
            return
        self.sim_model.eq_active0[self.weld_eq_id] = 1
        self.sim_model.eq_active0[self.base_weld_eq_id] = 1
        super().reset()
        self.data.eq_active[self.weld_eq_id] = 1
        self.data.eq_active[self.base_weld_eq_id] = 1
        self.release_time_s = None
        self.forced_release = False
        self.release_reason = None
        self.release_error = None
        self.release_pose = None
        self.release_plus_post_pose = None
        self.release_gripper_z = None
        self.release_block_gripper_distance_m = None
        self.post_release_block_gripper_distance_m = None
        self.physics_log = []
        self.release_transitions = 0
        self._elapsed_s = 0.0
        self._post_hook_inner = True
        self._initial_base_pose = self.data.qpos[
            self.body_pose_idx : self.body_pose_idx + 7
        ].copy()
        self._initial_relative_pose = self._relative_weld_pose()
        self._record_physics_step()

    def task_to_sim_ctrl(self, controls: np.ndarray) -> np.ndarray:
        """Clamp all sampled base velocity commands to zero."""
        controls = np.asarray(controls).copy()
        controls[..., :3] = 0.0
        output = super().task_to_sim_ctrl(controls)
        output[..., :3] = 0.0
        return output

    def _goal(self) -> np.ndarray:
        return np.array(
            [self.config.goal_x, self.config.goal_y, self.config.goal_z]
        )

    def _yaw_error_deg(self, quat: np.ndarray) -> float:
        quat = np.asarray(quat, dtype=float)
        yaw = np.degrees(
            np.arctan2(
                2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
            )
        )
        return float(
            abs((yaw - self.config.goal_yaw_deg + 180.0) % 360.0 - 180.0)
        )

    def _release_metrics(self) -> dict[str, float]:
        pose = self.data.qpos[
            self.object_pose_idx : self.object_pose_idx + 7
        ]
        delta = pose[:3] - self._goal()
        speed = np.linalg.norm(
            self.data.qvel[self.object_vel_idx : self.object_vel_idx + 3]
        )
        return {
            "xy_error_m": float(np.linalg.norm(delta[:2])),
            "z_error_m": float(delta[2]),
            "speed_m_s": float(speed),
            "yaw_error_deg": self._yaw_error_deg(pose[3:7]),
        }

    def _release_predicate(self, error: dict[str, float]) -> bool:
        return bool(
            error["xy_error_m"] <= self.config.release_xy_tolerance_m
            and self.config.release_z_lower_m
            <= error["z_error_m"]
            <= self.config.release_z_upper_m
            and error["speed_m_s"]
            <= self.config.release_speed_tolerance_m_s
            and error["yaw_error_deg"]
            <= self.config.release_yaw_tolerance_deg
        )

    def _release(self, forced: bool, error: dict[str, float]) -> None:
        if self.release_time_s is not None:
            return
        self.data.eq_active[self.weld_eq_id] = 0
        self.sim_model.eq_active0[self.weld_eq_id] = 0
        self.release_transitions += 1
        self.release_time_s = self._elapsed_s
        self.forced_release = forced
        self.release_reason = "forced_time" if forced else "predicate"
        self.release_error = error
        self.release_pose = self.data.qpos[
            self.object_pose_idx : self.object_pose_idx + 7
        ].tolist()
        self.release_gripper_z = float(
            self.data.site_xpos[self.finger_site_id, 2]
        )
        self.release_block_gripper_distance_m = float(
            np.linalg.norm(
                self.data.site_xpos[self.finger_site_id]
                - self.data.qpos[
                    self.object_pose_idx : self.object_pose_idx + 3
                ]
            )
        )

    def _project_active_weld(self) -> None:
        """Remove the native solver's sub-millimetre equality residual."""
        if not self.data.eq_active[self.weld_eq_id]:
            return
        finger_pos = self.data.site_xpos[self.finger_site_id].copy()
        finger_quat = np.empty(4)
        mujoco.mju_mat2Quat(
            finger_quat, self.data.site_xmat[self.finger_site_id]
        )
        offset = np.empty(3)
        mujoco.mju_rotVecQuat(
            offset, np.array([0.0, 0.0, 0.08]), finger_quat
        )
        self.data.qpos[
            self.object_pose_idx : self.object_pose_idx + 3
        ] = finger_pos - offset
        self.data.qpos[
            self.object_pose_idx + 3 : self.object_pose_idx + 7
        ] = finger_quat
        mujoco.mj_forward(self.sim_model, self.data)

    def post_sim_step(self) -> None:
        # The hierarchical simulator invokes this hook once before and once
        # after copying each native step back. Count only the latter.
        if self._post_hook_inner:
            self._post_hook_inner = False
            return
        self._post_hook_inner = True
        self._elapsed_s += float(self.sim_model.opt.timestep)
        self._project_active_weld()
        error = self._release_metrics()
        if self.release_time_s is None:
            if self._release_predicate(error):
                self._release(False, error)
            elif self._elapsed_s + 1e-12 >= self.config.forced_release_s:
                self._release(True, error)
        elif (
            self.release_plus_post_pose is None
            and self._elapsed_s
            >= self.release_time_s + self.config.post_release_s - 1e-12
        ):
            self.release_plus_post_pose = self.data.qpos[
                self.object_pose_idx : self.object_pose_idx + 7
            ].tolist()
            self.post_release_block_gripper_distance_m = float(
                np.linalg.norm(
                    self.data.site_xpos[self.finger_site_id]
                    - self.data.qpos[
                        self.object_pose_idx : self.object_pose_idx + 3
                    ]
                )
            )
        self._record_physics_step()

    def get_sim_metadata(self) -> dict[str, Any]:
        return {
            "weld_active": bool(self.data.eq_active[self.weld_eq_id]),
            "release_gripper_z": self.release_gripper_z,
        }

    def _relative_weld_pose(self) -> tuple[np.ndarray, np.ndarray]:
        finger_pos = self.data.site_xpos[self.finger_site_id].copy()
        box_pos = self.data.site_xpos[self.box_weld_site_id].copy()
        finger_quat = np.empty(4)
        box_quat = np.empty(4)
        mujoco.mju_mat2Quat(
            finger_quat, self.data.site_xmat[self.finger_site_id]
        )
        mujoco.mju_mat2Quat(
            box_quat, self.data.site_xmat[self.box_weld_site_id]
        )
        rel_pos = box_pos - finger_pos
        inv = np.array(
            [finger_quat[0], -finger_quat[1], -finger_quat[2], -finger_quat[3]]
        )
        rel_quat = np.empty(4)
        mujoco.mju_mulQuat(rel_quat, inv, box_quat)
        return rel_pos, rel_quat

    @staticmethod
    def _quat_error_deg(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float) / np.linalg.norm(a)
        b = np.asarray(b, dtype=float) / np.linalg.norm(b)
        return float(
            np.degrees(
                2.0 * np.arccos(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
            )
        )

    def _contact_counts(self) -> tuple[int, int]:
        robot_block = 0
        block_floor = 0
        box_geom = int(self.sim_model.geom("box_collision").id)
        for contact in self.data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if box_geom not in pair:
                continue
            other = next(iter(pair - {box_geom}), box_geom)
            other_body = int(self.sim_model.geom_bodyid[other])
            if other == self.ground_geom_id or other_body == 0:
                block_floor += 1
            elif other_body != self.box_body_id:
                robot_block += 1
        return robot_block, block_floor

    def _record_physics_step(self) -> None:
        robot_block, block_floor = self._contact_counts()
        rel_pos, rel_quat = self._relative_weld_pose()
        assert self._initial_relative_pose is not None
        initial_pos, initial_quat = self._initial_relative_pose
        base = self.data.qpos[
            self.body_pose_idx : self.body_pose_idx + 7
        ].copy()
        assert self._initial_base_pose is not None
        block = self.data.qpos[
            self.object_pose_idx : self.object_pose_idx + 7
        ].copy()
        self.physics_log.append(
            {
                "time_s": round(self._elapsed_s, 6),
                "weld_active": bool(
                    self.data.eq_active[self.weld_eq_id]
                ),
                "weld_position_drift_m": float(
                    np.linalg.norm(rel_pos - initial_pos)
                ),
                "weld_angle_drift_deg": self._quat_error_deg(
                    rel_quat, initial_quat
                ),
                "block_pose": block.tolist(),
                "block_speed_m_s": float(
                    np.linalg.norm(
                        self.data.qvel[
                            self.object_vel_idx : self.object_vel_idx + 3
                        ]
                    )
                ),
                "base_xy_displacement_m": float(
                    np.linalg.norm(base[:2] - self._initial_base_pose[:2])
                ),
                "base_yaw_change_deg": self._quat_error_deg(
                    base[3:7], self._initial_base_pose[3:7]
                ),
                "robot_block_contacts": robot_block,
                "block_floor_contacts": block_floor,
            }
        )

    def collision_manifest(self) -> dict[str, Any]:
        box_geom = int(self.sim_model.geom("box_collision").id)
        robot_geom_ids = [
            geom_id
            for geom_id in range(self.sim_model.ngeom)
            if int(self.sim_model.geom_bodyid[geom_id])
            not in (0, self.box_body_id)
            and (
                self.sim_model.geom_contype[geom_id]
                or self.sim_model.geom_conaffinity[geom_id]
            )
        ]
        excluded = all(
            not (
                self.sim_model.geom_contype[box_geom]
                & self.sim_model.geom_conaffinity[geom_id]
            )
            and not (
                self.sim_model.geom_conaffinity[box_geom]
                & self.sim_model.geom_contype[geom_id]
            )
            for geom_id in robot_geom_ids
        )
        floor_collides = bool(
            (
                self.sim_model.geom_contype[box_geom]
                & self.sim_model.geom_conaffinity[self.ground_geom_id]
            )
            or (
                self.sim_model.geom_conaffinity[box_geom]
                & self.sim_model.geom_contype[self.ground_geom_id]
            )
        )
        return {
            "robot_geom_count": len(robot_geom_ids),
            "all_robot_block_pairs_excluded": excluded,
            "block_floor_collision_enabled": floor_collides,
        }

    def runtime_idealization_manifest(self) -> list[str]:
        collision = self.collision_manifest()
        flags = {
            "oracle_structure_spec": True,
            "oracle_build_order": True,
            "oracle_block_pose": hasattr(self, "object_pose_idx"),
            "weld_grasp": self.weld_eq_id >= 0,
            "no_arm_block_collision": collision[
                "all_robot_block_pairs_excluded"
            ],
            "fixed_base": True,
            "sumo_to_assembly_state_bridge": True,
            "oracle_pick": True,
            "scripted_release_trigger": True,
            "sumo_box_proxy_dynamics": True,
        }
        return [name for name in IDEALIZATIONS if flags[name]]

    def cost_terms(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        """Return each H-018 rollout cost separately for logging."""
        batch_size = states.shape[0]
        qpos = states[..., : self.model.nq]
        qvel = states[..., self.model.nq :]
        placed = qpos[..., self.object_pose_idx : self.object_pose_idx + 3]
        delta = placed - self._goal()
        distance = np.linalg.norm(delta, axis=-1)
        xy_error = np.linalg.norm(delta[..., :2], axis=-1)

        placed_quat = qpos[
            ..., self.object_pose_idx + 3 : self.object_pose_idx + 7
        ]
        half_yaw = np.radians(self.config.goal_yaw_deg) / 2.0
        goal_quat = np.array(
            [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]
        )
        quat_norm = np.maximum(
            np.linalg.norm(placed_quat, axis=-1, keepdims=True), 1e-12
        )
        normalized_quat = placed_quat / quat_norm
        orientation_error = 1.0 - np.square(
            np.sum(normalized_quat * goal_quat, axis=-1)
        )
        block_z_dot_world_z = 1.0 - 2.0 * (
            np.square(normalized_quat[..., 1])
            + np.square(normalized_quat[..., 2])
        )
        placed_velocity = qvel[
            ..., self.object_vel_idx : self.object_vel_idx + 6
        ]
        base_height = qpos[..., self.body_pose_idx + 2]
        base_xy = qpos[..., self.body_pose_idx : self.body_pose_idx + 2]
        collision_clearance = np.linalg.norm(
            base_xy - placed[..., :2], axis=-1
        )
        gripper = sensors[
            ..., self.gripper_pos_idx : self.gripper_pos_idx + 3
        ]
        terminal_start = max(0, int(distance.shape[-1] * 0.75))
        terminal = distance[..., terminal_start:].mean(axis=-1)
        hover_deficit = np.maximum(
            0.0,
            self.config.goal_z + self.config.hover_clearance_m
            - placed[..., 2],
        )
        hover = np.where(
            xy_error > self.config.hover_xy_threshold_m,
            hover_deficit,
            0.0,
        )
        metadata = system_metadata or {}
        release_z = metadata.get("release_gripper_z")
        if metadata.get("weld_active", True) or release_z is None:
            retreat = np.zeros(batch_size)
        else:
            retreat = -self.config.w_retreat * np.maximum(
                0.0, gripper[..., 2] - float(release_z)
            ).mean(axis=-1)

        terms = {
            "place_linear": self.config.w_place_position
            * distance.mean(axis=-1),
            "place_quadratic": self.config.w_place_quadratic
            * np.square(distance).mean(axis=-1),
            "terminal_place": self.config.w_terminal_place * terminal,
            "place_orientation": self.config.w_place_orientation
            * orientation_error.mean(axis=-1),
            "upright": self.config.w_upright
            * (1.0 - np.clip(block_z_dot_world_z, -1.0, 1.0)).mean(
                axis=-1
            ),
            "collision": self.config.w_collision
            * (
                np.square(
                    np.maximum(0.0, 0.32 - collision_clearance)
                )
                + np.square(
                    np.maximum(
                        0.0,
                        self.config.object_half_height - placed[..., 2],
                    )
                )
            ).mean(axis=-1),
            "object_settle": self.config.w_object_settle
            * np.linalg.norm(placed_velocity, axis=-1).mean(axis=-1),
            "object_angular_velocity": self.config.w_object_angular_velocity
            * np.linalg.norm(placed_velocity[..., 3:], axis=-1).mean(
                axis=-1
            ),
            "base_stability": self.config.w_base_stability
            * np.square(base_height - 0.46).mean(axis=-1),
            "action_rate": self.config.w_action_rate
            * np.square(controls).sum(axis=-1).mean(axis=-1),
            "hover": self.config.w_hover * hover.mean(axis=-1),
            "retreat": retreat,
            # Frozen zero-weight terms remain explicit in every log.
            "gripper_object": np.zeros(batch_size),
            "gripper_height": np.zeros(batch_size),
        }
        return terms

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        terms = self.cost_terms(states, sensors, controls, system_metadata)
        total = -sum(terms.values())
        assert total.shape == (states.shape[0],)
        return total

    def success(
        self,
        model: MjModel,
        data: MjData,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        del model, metadata
        return bool(
            self.release_time_s is not None
            and self.release_plus_post_pose is not None
            and data.qpos[self.body_pose_idx + 2]
            > self.config.spot_fallen_threshold
        )

    def failure(
        self,
        model: MjModel,
        data: MjData,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        del model, metadata
        return bool(
            data.qpos[self.body_pose_idx + 2]
            <= self.config.spot_fallen_threshold
        )


register_task(
    SpotBlockMPCWeld.name,
    SpotBlockMPCWeld,
    SpotBlockMPCWeldConfig,
    **_SPOT_REGISTRATION_KWARGS,
)
set_config_overrides(
    SpotBlockMPCWeld.name,
    ControllerConfig,
    {"horizon": 2.0},
)
