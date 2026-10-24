"""Math helpers."""

import mujoco
import numpy as np


def body_rotation(quat_wxyz):
    """3x3 rotation matrix body -> world (its transpose maps world -> body)."""
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(quat_wxyz, dtype=np.float64))
    return rot.reshape(3, 3)


def rpm_to_action(rpm, hover_rpm, max_rpm):
    """Motor RPMs -> normalized action in [-1, 1] (-1 = off, 0 = hover, 1 = max)."""
    action = np.where(rpm <= hover_rpm, rpm / hover_rpm - 1, (rpm - hover_rpm) / (max_rpm - hover_rpm))
    return np.clip(action, -1, 1).astype(np.float32)


def yaw_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def gate_yaws(gates):
    """Gate normal yaw. Single gate: bisector of the arrival and departure directions, so the turn is
    split before and after it. Stacked pair (reverser): both gates share one plane, facing the
    bisector of the arrival and the reversed departure direction; the second gate is turned by 180 deg,
    so both are flown front to back."""
    n = len(gates)

    def partner(i, step):
        return np.linalg.norm(gates[(i + step) % n, :2] - gates[i, :2]) < 1e-6

    def neighbour(i, step):
        for k in range(1, n):
            j = (i + step * k) % n
            if np.linalg.norm(gates[j, :2] - gates[i, :2]) > 1e-6:
                return gates[j, :2]
        return gates[i, :2] + [step, 0.0]

    def unit(v):
        return v / max(np.linalg.norm(v), 1e-9)

    yaws = np.zeros(n)
    for i in range(n):
        arrive = unit(gates[i, :2] - neighbour(i, -1))
        leave = unit(neighbour(i, 1) - gates[i, :2])
        if partner(i, 1):           # first gate of a pair: in from `arrive`, back out along `leave`
            d = arrive - unit(neighbour(i + 1, 1) - gates[i, :2])
        elif partner(i, -1):        # second gate of a pair: same plane, opposite normal
            d = -(unit(gates[i, :2] - neighbour(i - 1, -1)) - leave)
        else:
            d = arrive + leave if np.linalg.norm(arrive + leave) > 1e-6 else arrive
        yaws[i] = np.arctan2(d[1], d[0])
    return yaws


def gate_corners(gates, yaws, half):
    """(N, 4, 3) world positions of the square corners at +-half (y, z) in each gate's plane."""
    local = np.array([[0, s * half, t * half] for s in (1, -1) for t in (1, -1)])
    return gates[:, None, :] + np.einsum("nij,kj->nki", np.stack([yaw_matrix(y) for y in yaws]), local)


def to_gate_frame(point, gates, yaws):
    """World point -> (N, 3) coordinates in every gate's frame: x along the normal, y sideways, z up."""
    d = point - gates
    c, s = np.cos(yaws), np.sin(yaws)
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]], axis=1)