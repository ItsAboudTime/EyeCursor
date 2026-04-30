from __future__ import annotations

import math
from typing import List

from panda3d.core import (
    AmbientLight,
    CardMaker,
    DirectionalLight,
    LineSegs,
    NodePath,
    SamplerState,
    TextureStage,
    Vec3,
    Vec4,
)

from game.core import asset_gen, settings
from game.core.camera_controller import CameraController
from game.core.horse_spawner import spawn_horses
from game.core.photo_manager import PhotoManager, PhotoState
from game.core.track import OvalTrack
from game.scenes.base_scene import BaseScene


GROUND_SIZE = 200.0
RETICLE_HALF = 0.018
RETICLE_GAP = 0.006
FILL_RING_RADIUS = 0.045
FILL_RING_SEGMENTS = 48


class GameScene(BaseScene):
    name = "game"
    cursor_visible = False

    def __init__(self, app) -> None:
        super().__init__(app)
        self.world: NodePath | None = None
        self.cart: NodePath | None = None
        self.camera_pivot: NodePath | None = None
        self.hud_root: NodePath | None = None
        self.fill_ring_node: NodePath | None = None
        self.track: OvalTrack | None = None
        self.t_param: float = 0.0
        self.speed: float = 3.0
        self.cam_ctrl: CameraController | None = None
        self.photo: PhotoManager | None = None
        self._lights: list = []
        self._accept_keys: list = []
        self._has_camera_state = False
        self._saved_cam_parent = None

    def enter(self) -> None:
        self.frozen = False
        base = self.app.base

        self.world = NodePath("game_world")
        self.world.reparentTo(base.render)

        self._build_environment()

        self.track = OvalTrack(a=30.0, b=18.0)
        self.cart = NodePath("cart")
        self.cart.reparentTo(self.world)

        self.camera_pivot = NodePath("camera_pivot")
        self.camera_pivot.reparentTo(self.cart)
        self.camera_pivot.setPos(0, 0, 0)

        self._saved_cam_parent = base.camera.getParent()
        base.camera.reparentTo(self.camera_pivot)
        base.camera.setPos(0, 0, 0)
        base.camera.setHpr(0, 0, 0)
        base.camLens.setFov(90)
        base.camLens.setNear(0.2)
        base.camLens.setFar(500)

        self.cam_ctrl = CameraController(base, self.camera_pivot)

        spawn_horses(base.loader, self.world, self.track.a, self.track.b, count=10)

        self._build_hud()

        speed_name = self.app.config.get("cart_speed", "normal")
        self.speed = settings.SPEED_VALUES.get(speed_name, 3.0)
        self.t_param = 0.0

        countdown = float(self.app.config.get("countdown_duration", 1.0))
        self.photo = PhotoManager(
            base=base,
            countdown_duration=countdown,
            get_depth=self.app.depth_client.get_depth,
            hud_root=self.hud_root,
            progress_setter=self._update_fill_ring,
            get_paused=lambda: self.frozen,
        )

        self._bind_inputs()
        self._has_camera_state = True

    def _build_environment(self) -> None:
        base = self.app.base

        cm = CardMaker("ground")
        cm.setFrame(-GROUND_SIZE / 2, GROUND_SIZE / 2, -GROUND_SIZE / 2, GROUND_SIZE / 2)
        ground = self.world.attachNewNode(cm.generate())
        ground.setP(-90)
        ground.setZ(0)
        gp = asset_gen.grass_path()
        if gp is not None:
            try:
                tex = base.loader.loadTexture(str(gp))
                tex.setMagfilter(SamplerState.FT_nearest)
                tex.setMinfilter(SamplerState.FT_nearest)
                tex.setWrapU(SamplerState.WM_repeat)
                tex.setWrapV(SamplerState.WM_repeat)
                ground.setTexture(tex)
                ground.setTexScale(TextureStage.getDefault(), 30, 30)
            except Exception:
                ground.setColor(Vec4(0.20, 0.55, 0.25, 1.0))
        else:
            ground.setColor(Vec4(0.20, 0.55, 0.25, 1.0))

        sky_cm = CardMaker("sky")
        sky_cm.setFrame(-1, 1, -1, 1)
        sky = base.render2dp.attachNewNode(sky_cm.generate())
        sky.setColor(Vec4(0.5, 0.75, 0.95, 1.0))
        base.cam2dp.node().getDisplayRegion(0).setSort(-20)
        sp = asset_gen.sky_path()
        if sp is not None:
            try:
                stex = base.loader.loadTexture(str(sp))
                stex.setMagfilter(SamplerState.FT_nearest)
                stex.setMinfilter(SamplerState.FT_nearest)
                sky.setTexture(stex)
            except Exception:
                pass
        self._sky_card = sky

        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.55, 0.55, 0.60, 1))
        amb_np = self.world.attachNewNode(ambient)
        self.world.setLight(amb_np)
        self._lights.append(amb_np)

        directional = DirectionalLight("directional")
        directional.setColor(Vec4(0.85, 0.82, 0.75, 1))
        dir_np = self.world.attachNewNode(directional)
        dir_np.setHpr(45, -45, 0)
        self.world.setLight(dir_np)
        self._lights.append(dir_np)

    def _build_hud(self) -> None:
        base = self.app.base
        self.hud_root = NodePath("hud_root")
        self.hud_root.reparentTo(base.aspect2d)

        shadow = self._make_reticle(Vec4(0, 0, 0, 1), shift=0.0035)
        shadow.reparentTo(self.hud_root)
        cross = self._make_reticle(Vec4(1, 1, 1, 1), shift=0.0)
        cross.reparentTo(self.hud_root)

        self.fill_ring_node = NodePath("fill_ring")
        self.fill_ring_node.reparentTo(self.hud_root)
        self.fill_ring_node.hide()

    def _make_reticle(self, color: Vec4, shift: float) -> NodePath:
        ls = LineSegs()
        ls.setThickness(2.0)
        ls.setColor(color)
        sx, sy = shift, -shift
        ls.moveTo(sx + RETICLE_GAP, 0, sy)
        ls.drawTo(sx + RETICLE_HALF, 0, sy)
        ls.moveTo(sx - RETICLE_GAP, 0, sy)
        ls.drawTo(sx - RETICLE_HALF, 0, sy)
        ls.moveTo(sx, 0, sy + RETICLE_GAP)
        ls.drawTo(sx, 0, sy + RETICLE_HALF)
        ls.moveTo(sx, 0, sy - RETICLE_GAP)
        ls.drawTo(sx, 0, sy - RETICLE_HALF)
        node = ls.create()
        return NodePath(node)

    def _update_fill_ring(self, progress: float) -> None:
        if self.fill_ring_node is None:
            return
        for child in self.fill_ring_node.getChildren():
            child.removeNode()
        if progress <= 0.0:
            self.fill_ring_node.hide()
            return
        self.fill_ring_node.show()
        progress = max(0.0, min(1.0, progress))
        green = Vec4(0.30, 0.85, 0.30, 1.0)
        yellow = Vec4(0.95, 0.85, 0.20, 1.0)
        col = green + (yellow - green) * progress
        ls = LineSegs()
        ls.setThickness(3.0)
        ls.setColor(col)
        n = max(1, int(FILL_RING_SEGMENTS * progress))
        end_angle = 2.0 * math.pi * progress
        start_angle = -math.pi / 2.0
        for i in range(n + 1):
            a = start_angle + (end_angle * i / n)
            x = math.cos(a) * FILL_RING_RADIUS
            z = math.sin(a) * FILL_RING_RADIUS
            if i == 0:
                ls.moveTo(x, 0, z)
            else:
                ls.drawTo(x, 0, z)
        node = ls.create()
        NodePath(node).reparentTo(self.fill_ring_node)

    def _bind_inputs(self) -> None:
        base = self.app.base
        trigger = self.app.config.get("photo_trigger", "left_click")

        base.accept("escape", self._on_escape)
        self._accept_keys.append("escape")

        if trigger == "left_click":
            press, release = "mouse1", "mouse1-up"
        elif trigger == "right_click":
            press, release = "mouse3", "mouse3-up"
        else:
            press, release = "space", "space-up"

        base.accept(press, self._on_trigger_press)
        base.accept(release, self._on_trigger_release)
        self._accept_keys.extend([press, release])

    def _unbind_inputs(self) -> None:
        base = self.app.base
        for k in self._accept_keys:
            base.ignore(k)
        self._accept_keys = []

    def _on_escape(self) -> None:
        self.app.scene_manager.push_overlay("pause")

    def _on_trigger_press(self) -> None:
        if self.photo is not None and not self.frozen:
            self.photo.set_trigger_held(True)

    def _on_trigger_release(self) -> None:
        if self.photo is not None:
            self.photo.set_trigger_held(False)

    def update(self, dt: float) -> None:
        if self.frozen:
            return
        if self.track is None or self.cart is None:
            return
        cart_frozen = self.photo is not None and self.photo.state in (
            PhotoState.HOLDING,
            PhotoState.FLASHING,
        )
        if not cart_frozen:
            self.t_param = self.track.advance(self.t_param, self.speed, dt)
        pos, tangent = self.track.evaluate(self.t_param)
        self.cart.setPos(pos)
        self.cart.lookAt(pos + tangent)
        if self.cam_ctrl is not None and not cart_frozen:
            self.cam_ctrl.update(dt)
        if self.photo is not None:
            self.photo.tick(dt)

    def exit(self) -> None:
        self._unbind_inputs()
        if self.photo is not None:
            self.photo.cleanup()
            self.photo = None

        base = self.app.base
        if self._has_camera_state and self._saved_cam_parent is not None:
            base.camera.reparentTo(self._saved_cam_parent)
            base.camera.setPos(0, 0, 0)
            base.camera.setHpr(0, 0, 0)
            base.camLens.setFov(90)
            self._has_camera_state = False
            self._saved_cam_parent = None

        if self.hud_root is not None:
            self.hud_root.removeNode()
            self.hud_root = None
        if hasattr(self, "_sky_card") and self._sky_card is not None:
            self._sky_card.removeNode()
            self._sky_card = None
        if self.world is not None:
            self.world.removeNode()
            self.world = None
        self.cart = None
        self.camera_pivot = None
        self.fill_ring_node = None
        self.track = None
        self._lights = []
