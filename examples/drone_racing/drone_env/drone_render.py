"""Offscreen MuJoCo rendering of square gates and a fading flight trace; optional OpenCV window."""

import time
from collections import deque

import mujoco
import numpy as np

from .drone_constants import GATE_SIZE, GATE_HOLE, GATE_DEPTH
from .drone_utils import yaw_matrix

CAMERAS = ("third_person", "track", "course")

SKY_BLUE, ROYAL_BLUE, VIOLET = (0.53, 0.81, 0.98), (0.25, 0.41, 0.88), (0.54, 0.17, 0.89)
BLACK, WHITE = (0.05, 0.05, 0.08), (1.0, 1.0, 1.0)

# gate colors: (top/bottom bars, side bars)
GATE_NEXT, GATE_FUTURE = (WHITE, SKY_BLUE), (SKY_BLUE, ROYAL_BLUE)
GATE_OTHER = ((ROYAL_BLUE, BLACK), (VIOLET, BLACK))   # alternates along the course

TRACE_LEN, TRACE_MAX_SPEED = 300, 4.0                 # 6 s at 50 Hz; speed shown as full violet

_BAR, _OFF = (GATE_SIZE - GATE_HOLE) / 2, (GATE_SIZE + GATE_HOLE) / 4
_BARS = [((0, 0, _OFF), (GATE_DEPTH / 2, GATE_SIZE / 2, _BAR / 2), 0),     # top    (local pos, half size, color)
         ((0, 0, -_OFF), (GATE_DEPTH / 2, GATE_SIZE / 2, _BAR / 2), 0),    # bottom
         ((0, _OFF, 0), (GATE_DEPTH / 2, _BAR / 2, GATE_HOLE / 2), 1),     # left
         ((0, -_OFF, 0), (GATE_DEPTH / 2, _BAR / 2, GATE_HOLE / 2), 1)]    # right


class DroneRenderer:
    def __init__(self, model, width=640, height=480, window="drone race", trace=False):
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.scene = self.renderer.scene
        self.window = window
        self.trace = deque(maxlen=TRACE_LEN) if trace else None
        self.reset()

    def reset(self):
        self.wall_start = None
        self.azimuth = None
        if self.trace is not None:
            self.trace.clear()

    def render(self, data, pos, speed, gates, yaws, highlight, camera="third_person"):
        """RGB frame. `highlight` = gate indices in the observation, next gate first."""
        self.renderer.update_scene(data, self._camera(camera, pos, gates[highlight[0]], gates))
        self._draw_gates(gates, yaws, highlight)
        if self.trace is not None:
            self.trace.append((np.array(pos), speed))
            self._draw_trace()
        return self.renderer.render()

    def _draw_gates(self, gates, yaws, highlight):
        for i, (gate, yaw) in enumerate(zip(gates, yaws)):
            colors = GATE_NEXT if i == highlight[0] else GATE_FUTURE if i in highlight[1:] else GATE_OTHER[i % 2]
            rot = yaw_matrix(yaw)
            for k, (local, half, c) in enumerate(_BARS):
                geom = self._geom(mujoco.mjtGeom.mjGEOM_BOX, half, gate + rot @ local, (*colors[c], 1.0), rot)
                if k == 0:
                    geom.label = str(i + 1)  # flight order, on the top bar
        for a, b in zip(gates, np.roll(gates, -1, axis=0)):
            self._line(a, b, (*WHITE, 0.15), 1.0)

    def _draw_trace(self):
        n = len(self.trace)
        for k in range(1, n):
            t = min(self.trace[k][1] / TRACE_MAX_SPEED, 1.0)
            rgb = (1 - t) * np.array(SKY_BLUE) + t * np.array(VIOLET)          # slow -> fast
            self._line(self.trace[k - 1][0], self.trace[k][0], (*rgb, 0.95 * (k / n) ** 1.5), 3.0)

    def _camera(self, camera, pos, next_gate, gates):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        if camera == "third_person":  # behind the drone, facing the next gate
            to_gate = next_gate - pos
            target = np.degrees(np.arctan2(to_gate[1], to_gate[0]))
            if self.azimuth is None:
                self.azimuth = target
            self.azimuth += 0.1 * ((target - self.azimuth + 180) % 360 - 180)  # smooth turns
            cam.lookat[:] = pos + 0.3 * to_gate
            cam.distance = 1.2 + 0.3 * np.linalg.norm(to_gate)
            cam.azimuth, cam.elevation = self.azimuth, -12
        elif camera == "track":       # follows the drone from a fixed side angle
            cam.lookat[:] = pos
            cam.distance, cam.azimuth, cam.elevation = 2.0, -45, -20
        else:                         # course overview, framed on the gates
            cam.lookat[:] = (gates.max(0) + gates.min(0)) / 2
            cam.distance = max(2.0, 1.3 * np.linalg.norm(gates.max(0) - gates.min(0)))
            cam.azimuth, cam.elevation = -60, -40
        return cam

    def _line(self, a, b, rgba, width):
        geom = self._geom(mujoco.mjtGeom.mjGEOM_LINE, (0, 0, 0), (0, 0, 0), rgba)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(a, float), np.asarray(b, float))

    def _geom(self, geom_type, size, pos, rgba, rot=np.eye(3)):
        geom = self.scene.geoms[self.scene.ngeom]
        mujoco.mjv_initGeom(geom, geom_type, np.asarray(size, float), np.asarray(pos, float),
                            rot.ravel(), np.asarray(rgba, np.float32))
        self.scene.ngeom += 1
        return geom

    def show(self, frame, sim_time, lines=()):
        """Show a frame in an OpenCV window at real-time speed; `lines` = [(text, scale)]. Returns the key or ''."""
        import cv2
        now = time.perf_counter()
        if self.wall_start is None:
            self.wall_start = now - sim_time
        time.sleep(max(0.0, sim_time - (now - self.wall_start)))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for k, (text, scale) in enumerate(lines):
            for color, thick in (((0, 0, 0), 4), ((255, 255, 255), 1)):   # outlined text
                cv2.putText(frame, text, (12, 30 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        cv2.imshow(self.window, frame)
        key = cv2.waitKey(1)
        return chr(key & 0xFF) if key != -1 else ""

    def close(self):
        self.renderer.close()
        try:
            import cv2
            cv2.destroyWindow(self.window)
            cv2.waitKey(1)
        except Exception:  # window never opened / no OpenCV
            pass