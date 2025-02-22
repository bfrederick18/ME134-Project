import rclpy
import numpy as np

from math import pi


LOGGER_LEVEL = rclpy.logging.LoggingSeverity.INFO  # rclpy.logging.LoggingSeverity.DEBUG or rclpy.logging.LoggingSeverity.INFO

TRAJ_RATE = 200  # Hertz

CYCLE = 2 * pi
WAITING_POS = [-pi / 2, 0.0, -pi / 2, 0.0, 0.0]

JOINT_NAMES = ['base', 'shoulder', 'elbow', 'wrist', 'gripper']

HSV_LIMITS_PURPLE = np.array([[170, 190], [90, 140], [40, 120]])