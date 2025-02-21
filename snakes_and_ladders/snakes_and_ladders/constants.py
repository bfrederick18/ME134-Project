from math import pi
import numpy as np


TRAJ_RATE = 100  # Hertz

CYCLE = 2 * pi
WAITING_POS = [0.0, 0.0, -pi / 2, 0.0]

JOINT_NAMES = ["base", "shoulder", "elbow", "wrist"]

HSV_LIMITS_PURPLE = np.array([[170, 190], [90, 140], [70, 120]])