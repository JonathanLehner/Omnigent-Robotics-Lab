"""Assembly world in plain MuJoCo: Spot-with-arm (Menagerie), handled blocks, IK, idealized grasp.

Idealizations are explicit flags so every result can list which ones were in use:
  fixed_base           robot placed at the build site, base welded to the world (rung 2 replaces with walking)
  weld_grasp           a grasped block rigidly follows the gripper (rung 3 replaces with a real gripper grasp)
  arm_block_collision  False disables arm-vs-block contacts (only meaningful together with weld_grasp)
"""

from pathlib import Path

import mujoco
import numpy as np
_CACHED_SPOT = Path.home() / ".cache/robot_descriptions/mujoco_menagerie/boston_dynamics_spot/spot_arm.xml"
if _CACHED_SPOT.exists():
    # Importing robot_descriptions runs `git checkout` on the shared cache; 8 parallel workers race on its
    # index.lock and crash, so use the cached file directly once it exists.
    SPOT_XML = _CACHED_SPOT
else:
    from robot_descriptions import spot_mj_description

    SPOT_XML = Path(spot_mj_description.PACKAGE_PATH) / "spot_arm.xml"
SITE = np.array([0.85, 0.0, 0.0])  # build-site origin (structure frame) in world coordinates
BLOCK_TYPES = {"cube": (0.12, 0.12, 0.10), "brick": (0.12, 0.24, 0.10)}  # full x, y, z sizes (m)
BLOCK_MASS = {"cube": 0.4, "brick": 0.8}
COLORS = {
    "red": (0.85, 0.15, 0.15),
    "blue": (0.15, 0.35, 0.85),
    "green": (0.2, 0.7, 0.25),
    "yellow": (0.95, 0.8, 0.1),
    "orange": (0.95, 0.5, 0.1),
    "purple": (0.55, 0.25, 0.75),
    "cyan": (0.1, 0.75, 0.8),
    "white": (0.92, 0.92, 0.92),
}
HANDLE_HALF = (0.015, 0.035, 0.01)  # bar on the block's -x face, facing the robot
HANDLE_Z = 0.03  # handle sits above block center: the gripper cannot reach lower than ~7 cm without hitting the floor
ARM_JOINTS = ["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1"]
TIP = np.array([0.19, 0.0, 0.0])  # grasp point in the arm_link_wr1 frame (between the jaws)
GRASP_PITCH_DEG = (45, 60, 30)  # gripper pitch-down angles tried, in order, for a reachable grasp
READY = np.array([0.0, -1.0, 1.6, 0.0, 0.2, 0.0])  # arm reaching forward-down; second IK seed


def _home():
    """Joint name -> angle from Menagerie's 'home' keyframe."""
    m = mujoco.MjModel.from_xml_path(str(SPOT_XML))
    key = m.key("home").qpos
    return {m.joint(j).name: float(key[m.jnt_qposadr[j]]) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}


HOME = _home()


def handle_point(btype: str) -> np.ndarray:
    """Grasp point in the block frame: center of the handle bar."""
    return np.array([-(BLOCK_TYPES[btype][0] / 2 + HANDLE_HALF[0]), 0.0, HANDLE_Z])


def add_block(spec, bid, btype, color, pos, quat, collide_robot):
    sx, sy, sz = BLOCK_TYPES[btype]
    ct = 3 if collide_robot else 2  # robot geoms are contype/conaffinity 1, floor is 3
    body = spec.worldbody.add_body(name=bid, pos=list(pos), quat=list(quat))
    body.add_freejoint(name=f"{bid}_free")
    rgba = [*COLORS[color], 1]
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx / 2, sy / 2, sz / 2], rgba=rgba,
                  mass=BLOCK_MASS[btype], friction=[0.9, 0.01, 0.001], contype=ct, conaffinity=ct)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(HANDLE_HALF), pos=[-(sx / 2 + HANDLE_HALF[0]), 0, HANDLE_Z],
                  rgba=[0.2, 0.2, 0.2, 1], mass=0.02, contype=ct, conaffinity=ct)


def build_model(blocks: list[dict], robot: bool = True, arm_block_collision: bool = False, physics: dict | None = None) -> mujoco.MjModel:
    """blocks: [{id, type, color, pos (world), quat}]."""
    spec = mujoco.MjSpec.from_file(str(SPOT_XML)) if robot else mujoco.MjSpec()
    if robot:
        for key in list(spec.keys):
            spec.delete(key)
        spec.delete(spec.joint("freejoint"))  # fixed_base idealization
        spec.body("body").pos = [0, 0, 0.46]
        # A jointless body is welded to the world, which disables MuJoCo's parent-child contact filter.
        for child in ["arm_link_sh0", "fl_hip", "fr_hip", "hl_hip", "hr_hip"]:
            spec.add_exclude(bodyname1="body", bodyname2=child)
        for g in spec.geoms:
            if g.contype or g.conaffinity:
                g.contype, g.conaffinity = 1, 1
    spec.option.timestep = 0.002
    # physics: None = MuJoCo defaults (pyramidal cone, impratio 1), which let stacked/grasped blocks creep: a cube on
    # a cube with mu=0.9 slid 3.6 cm at 30 deg tilt. {"cone": "elliptic", "impratio": 10} (MuJoCo's advice for
    # manipulation) cut that ~50x. Opt-in per method config (`physics:`) so experiments stay comparable.
    if physics:
        if physics.get("cone") == "elliptic":
            spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = physics.get("impratio", 1)
        spec.option.noslip_iterations = physics.get("noslip_iterations", 0)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.82, 0.82, 0.8], rgb2=[0.74, 0.74, 0.72], width=512, height=512)
    spec.add_material(name="grid", textures=["", "grid"], texrepeat=[8, 8])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.1], material="grid", contype=3, conaffinity=3,
                            friction=[0.9, 0.01, 0.001])
    spec.worldbody.add_light(pos=[0.5, -1, 3], dir=[0, 0.3, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.7, 0.7, 0.7])
    spec.worldbody.add_light(pos=[1, 1, 2], dir=[-0.2, -0.3, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.3, 0.3, 0.3])
    for b in blocks:
        add_block(spec, b["id"], b["type"], b["color"], b["pos"], b["quat"], arm_block_collision or not robot)
    return spec.compile()


def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rot(q, v):
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.asarray(v, float), np.asarray(q, float))
    return out


def pitch_quat(deg):
    h = np.radians(deg) / 2
    return np.array([np.cos(h), 0.0, np.sin(h), 0.0])


class World:
    """One episode's physics. Units: meters, seconds; poses are world frame, quats wxyz."""

    def __init__(self, blocks, arm_block_collision=False, frame_every_s=None, cam=None, target=None, target_pictures=None, physics=None):
        """target: world-frame target blocks, drawn as ghosts in video frames (render only, physics unaffected).
        target_pictures: {label: png path} shown next to the live views in video frames."""
        self.target, self.target_pictures = target or [], target_pictures or {}
        self.m = build_model(blocks, robot=True, arm_block_collision=arm_block_collision, physics=physics)
        self.d = mujoco.MjData(self.m)
        self.dk = mujoco.MjData(self.m)  # scratch data for IK
        self.base_body = self.m.body("body").id
        self.wr1 = self.m.body("arm_link_wr1").id
        self.arm_q = [self.m.joint(j).qposadr[0] for j in ARM_JOINTS]
        self.arm_v = [self.m.joint(j).dofadr[0] for j in ARM_JOINTS]
        self.arm_lo = np.array([self.m.joint(j).range[0] for j in ARM_JOINTS])
        self.arm_hi = np.array([self.m.joint(j).range[1] for j in ARM_JOINTS])
        for j in range(self.m.njnt):
            if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                self.d.qpos[self.m.jnt_qposadr[j]] = HOME.get(self.m.joint(j).name, 0.0)
        for a in range(self.m.nu):
            self.d.ctrl[a] = HOME.get(self.m.joint(self.m.actuator_trnid[a, 0]).name, 0.0)
        mujoco.mj_forward(self.m, self.d)
        self.held = None  # (block id, rel pos, rel quat)
        self.frames, self.frame_every_s, self.cam = [], frame_every_s, cam
        self._renderer = None
        self._next_frame = 0.0
        # Optional read-only instrumentation hook. Stages may observe state
        # after each simulation step, but must not step the world themselves.
        self.step_observer = None

    # --- state -------------------------------------------------------------
    def block_pose(self, bid):
        a = self.m.joint(f"{bid}_free").qposadr[0]
        return self.d.qpos[a:a + 3].copy(), self.d.qpos[a + 3:a + 7].copy()

    def base_pose(self):
        return self.m.body_pos[self.base_body].copy(), self.m.body_quat[self.base_body].copy()

    def set_base_pose(self, pos, quat):
        """Set the welded assembly robot root; subsequent IK uses this exact pose."""
        self.m.body_pos[self.base_body] = np.asarray(pos, dtype=float)
        q = np.asarray(quat, dtype=float)
        self.m.body_quat[self.base_body] = q / np.linalg.norm(q)
        mujoco.mj_forward(self.m, self.d)

    def tip_pose(self, d=None):
        d = d or self.d
        q = d.xquat[self.wr1].copy()
        return d.xpos[self.wr1] + rot(q, TIP), q

    # --- control -----------------------------------------------------------
    def ik(self, pos, quat):
        """Best of two seeds (current arm targets, forward-reach posture); the folded home pose has local minima."""
        current = self.d.ctrl[[self._act(j) for j in ARM_JOINTS]].copy()
        sols = [self._ik(pos, quat, seed) for seed in (current, READY)]
        return min(sols, key=lambda s: s[1] + 0.01 * s[2])

    def _ik(self, pos, quat, seed, iters=200):
        """Damped least squares on the 6 arm joints. Returns (q, pos_residual_m, ang_residual_deg)."""
        m, d = self.m, self.dk
        d.qpos[:] = self.d.qpos
        q = np.array(seed, float)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            d.qpos[self.arm_q] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            tip, cq = self.tip_pose(d)
            ep = np.asarray(pos) - tip
            er = np.zeros(3)
            mujoco.mju_subQuat(er, np.asarray(quat, float), cq)
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
                break
            mujoco.mj_jac(m, d, jp, jr, tip, self.wr1)
            J = np.vstack([jp[:, self.arm_v], 0.5 * jr[:, self.arm_v]])
            e = np.concatenate([ep, 0.5 * er])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.arm_lo, self.arm_hi)
        return q, float(np.linalg.norm(ep)), float(np.degrees(np.linalg.norm(er)))

    def _act(self, joint):
        return next(a for a in range(self.m.nu) if self.m.actuator_trnid[a, 0] == self.m.joint(joint).id)

    def move_joints(self, q_target, duration=1.0, dwell=0.3):
        idx = [self._act(j) for j in ARM_JOINTS]
        q0 = self.d.ctrl[idx].copy()
        n = max(1, int(duration / self.m.opt.timestep))
        for i in range(n):
            s = (i + 1) / n
            self.d.ctrl[idx] = q0 + (3 * s * s - 2 * s ** 3) * (np.asarray(q_target) - q0)  # smoothstep
            self.step(1)
        self.step(int(dwell / self.m.opt.timestep))

    def set_gripper(self, open_):
        self.d.ctrl[self._act("arm_f1x")] = -1.0 if open_ else 0.0

    def grasp(self, bid):
        """Idealized weld: from now on the block rigidly follows the gripper."""
        tip, tq = self.tip_pose()
        bp, bq = self.block_pose(bid)
        self.held = (bid, rot(quat_conj(tq), bp - tip), quat_mul(quat_conj(tq), bq))

    def release(self):
        self.held = None

    def step(self, n):
        for _ in range(n):
            if self.held:
                bid, rp, rq = self.held
                tip, tq = self.tip_pose()
                a = self.m.joint(f"{bid}_free").qposadr[0]
                v = self.m.joint(f"{bid}_free").dofadr[0]
                self.d.qpos[a:a + 3] = tip + rot(tq, rp)
                self.d.qpos[a + 3:a + 7] = quat_mul(tq, rq)
                self.d.qvel[v:v + 6] = 0
            mujoco.mj_step(self.m, self.d)
            if self.step_observer is not None:
                self.step_observer()
            if self.frame_every_s and self.d.time >= self._next_frame:
                self.frames.append(self.render())
                self._next_frame += self.frame_every_s

    def render(self, w=320, h=240):
        """One video frame: VIDEO_VIEWS tiled 2x2 and labeled (640x480 total)."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.m, h, w)
            soften_lights(self.m)  # rendering-only fields, physics unaffected
        imgs = {}
        for name, view in VIDEO_VIEWS.items():
            cam = self.cam if (name == "main" and self.cam) else view_camera(view, SITE + [-0.25, 0, 0.2])
            self._renderer.update_scene(self.d, camera=cam)
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0  # shadows cost ~80% of render time
            self._draw_ghosts(self._renderer.scene)
            imgs[name] = self._renderer.render().copy()
        frame = tile_views(imgs)
        if self.target_pictures:  # third column: what Spot was asked to build
            from PIL import Image

            if not hasattr(self, "_target_col"):
                col = [np.asarray(Image.open(p).convert("RGB").resize((w, h))) for p in self.target_pictures.values()]
                self._target_col = tile_views(dict(zip(self.target_pictures, col)))[:, :w] if len(col) == 1 else \
                    np.concatenate([tile_views({k: c})[:h, :w] for k, c in zip(self.target_pictures, col)], 0)
            frame = np.concatenate([frame, self._target_col[:frame.shape[0]]], 1)
        return frame

    def _draw_ghosts(self, scn):
        """Semi-transparent target blocks, added to the rendered scene only."""
        for b in self.target:
            if scn.ngeom >= scn.maxgeom:
                return
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, np.asarray(b["quat"], float))
            sx, sy, sz = BLOCK_TYPES[b["type"]]
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.array([sx, sy, sz]) / 2,
                                np.asarray(b["pos"], float), mat, np.array([*COLORS[b["color"]], 0.28], np.float32))
            scn.ngeom += 1


# Named views (azimuth, elevation, distance). Azimuth 0 looks from the robot's side (+x), 180 from beyond the
# structure, 270 from the robot's left (+y). Target pictures use STRUCTURE_VIEWS; episode videos use VIDEO_VIEWS.
STRUCTURE_VIEWS = {"main": (200, -22, 1.1), "top": (180, -89.9, 0.9), "robot side": (0, -12, 1.1),
                   "left side": (270, -12, 1.1)}
VIDEO_VIEWS = {"main": (215, -28, 1.9), "top": (180, -89.9, 1.8), "far side": (180, -15, 1.6),
               "left side": (270, -12, 2.4)}


def soften_lights(m):
    """Rendering-only: dimmer lights, no specular. The defaults saturate up-facing surfaces (orange reads as yellow)."""
    m.light_specular[:] = 0
    m.light_diffuse[:] = m.light_diffuse * 0.6
    m.vis.headlight.specular[:] = 0
    m.vis.headlight.diffuse[:] = 0.24
    m.vis.headlight.ambient[:] = 0.15


def view_camera(view, lookat):
    az, el, dist = view
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.azimuth, cam.elevation, cam.distance = az, el, dist
    return cam


def tile_views(images: dict) -> np.ndarray:
    """2x2 grid of named views, each labeled in its corner."""
    from PIL import Image, ImageDraw

    tiles = []
    for name, img in images.items():
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, 8 + 7 * len(name), 16], fill=(0, 0, 0))
        dr.text((4, 3), name, fill=(255, 255, 255))
        tiles.append(np.asarray(im))
    while len(tiles) < 4:
        tiles.append(np.zeros_like(tiles[0]))
    return np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:4], 1)], 0)


def render_structure_views(blocks: list[dict], stem: str, w=480, h=360) -> dict:
    """Target pictures from STRUCTURE_VIEWS: <stem>_<view>.png each, plus <stem>_multiview.png (2x2, labeled)."""
    from PIL import Image

    m = build_model(blocks, robot=False)
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, 250)
    soften_lights(m)
    r = mujoco.Renderer(m, h, w)
    imgs, paths = {}, {}
    for name, view in STRUCTURE_VIEWS.items():
        r.update_scene(d, camera=view_camera(view, SITE + [0, 0, 0.12]))
        imgs[name] = r.render().copy()
        paths[name] = f"{stem}_{name.replace(' ', '_')}.png"
        Image.fromarray(imgs[name]).save(paths[name])
    paths["multiview"] = f"{stem}_multiview.png"
    Image.fromarray(tile_views(imgs)).save(paths["multiview"])
    return paths


def default_camera(azimuth=215.0, elevation=-28.0, distance=1.9):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = SITE + [-0.25, 0, 0.2]
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    return cam


def settle(blocks: list[dict], seconds=3.0) -> dict:
    """Settle test: simulate blocks alone (no robot). Returns max drift per block (m)."""
    m = build_model(blocks, robot=False)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    p0 = {b["id"]: d.body(b["id"]).xpos.copy() for b in blocks}
    mujoco.mj_step(m, d, int(seconds / m.opt.timestep))
    drift = {bid: float(np.linalg.norm(d.body(bid).xpos - p)) for bid, p in p0.items()}
    return {"stable": max(drift.values(), default=0) < 0.01, "drift_m": drift}


def render_structure(blocks: list[dict], path: str, cam=None, w=640, h=480):
    """Render blocks alone after settling briefly; this is how target pictures are made."""
    m = build_model(blocks, robot=False)
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, 250)
    r = mujoco.Renderer(m, h, w)
    if cam is None:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = SITE + [0, 0, 0.1]
        cam.azimuth, cam.elevation, cam.distance = 200.0, -22.0, 1.1
    r.update_scene(d, camera=cam)
    from PIL import Image

    Image.fromarray(r.render()).save(path)
    return path
