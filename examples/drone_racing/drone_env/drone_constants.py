"""Reward weights and task settings."""

# Reward
GATE_PASSED     = 1.0       # per gate
PROGRESS        = 1.0       # x metres moved toward the next gate centre, every step
ALIGN           = 1.0       # x change of gate alignment (0..1, 1 = on the gate's axis), every step
CRASH_PENALTY   = 2.0       # once, on crash (floor, flip or gate frame)
YAW_RATE_PENALTY= 0.001

# Episode   
EPISODE_LEN_SEC = 30.0
MAX_LAPS        = None      # None = only the time limit ends the episode
LOW_ALTITUDE    = 0.1       # m
MAX_TILT        = 2.0       # rad, |roll| or |pitch| above this = crash

# Gates: square frame with a square hole, normal along the flight path
GATE_SIZE       = 0.7       # m, outer edge
GATE_HOLE       = 0.5       # m, hole edge
GATE_DEPTH      = 0.05      # m, frame thickness
GATE_FORWARD_ONLY = False   # True: a gate counts only when crossed along its normal (front -> back)

# Drone
INITIAL_XYZ = (0.0, 0.0, 0.5)
DEFAULT_GATES = [                   # fixed oval used without a randomizer; clear of the start point
    [1.5, 0.0, 1.0], [3.0, 1.2, 1.2], [2.0, 2.6, 1.5],
    [0.0, 2.6, 1.3], [-1.5, 1.2, 1.0], [-0.5, -1.2, 0.8],
]