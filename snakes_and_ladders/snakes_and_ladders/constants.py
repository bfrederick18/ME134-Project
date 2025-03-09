import rclpy
import numpy as np

from math import pi


LOGGER_LEVEL = rclpy.logging.LoggingSeverity.DEBUG  # rclpy.logging.LoggingSeverity.DEBUG or rclpy.logging.LoggingSeverity.INFO

TRAJ_RATE = 200  # Hertz

CYCLE = 2 * pi
WAITING_POS = [-pi / 12 * 5, 0.0, -pi / 2, 0.0, 0.0]  # [0.0, 0.0, -pi / 2, 0.0, 0.0]

JOINT_NAMES = ['base', 'shoulder', 'elbow', 'wrist', 'gripper']

HSV_LIMITS_PURPLE = np.array([[119, 140], [62, 189], [81, 125]])
HSV_LIMITS_BLUE = np.array([[104, 129], [141, 224], [138, 180]])  # [[105, 109], [191, 222], [138, 161]]
HSV_LIMITS_DISH = np.array([[9, 14], [96, 169], [138, 196]])
HSV_LIMITS_SIDECAM = np.array([[0, 10], [50, 100], [150, 180]])

GRIPPER_INTERMEDIATE = -0.2
GRIPPER_CLOSE_PURPLE = -0.4
GRIPPER_CLOSE_DICE = -0.5