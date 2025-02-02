'''hw5p5.py

   This is the skeleton code for HW5 Problem 5.  Please EDIT.

   This creates a purely rotational movement.

'''

import rclpy
import numpy as np

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

# Grab the utilities
from hw5code.GeneratorNode      import GeneratorNode
from hw5code.TransformHelpers   import *
from hw5code.TrajectoryUtils    import *


#
#   Gimbal Kinematics
#
def fkin(q):
    return Rotz(q[0]) @ Rotx(q[1]) @ Roty(q[2])

def Jac(q):
    J1 = nz()
    J2 = Rotz(q[0]) @ nx()
    J3 = Rotz(q[0]) @ Rotx(q[1]) @ ny()
    return np.hstack((J1.reshape((3,1)), J2.reshape((3,1)), J3.reshape((3,1))))


#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node):
        # Initialize the current joint position to the starting
        # position and set the desired orientation to match.
        self.qd = np.zeros(3)
        self.Rd = Reye()

        # Pick the convergence bandwidth.
        self.lam = 20

    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names FOR THE EXPECTED URDF!
        return ['pan', 'tilt', 'roll']

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
        # Choose the alpha/beta angles based on the phase.
        if t <= 2.0:
            # Part A (t<=2):
            (alpha, alphadot) = ...  FIXME
            (beta,  betadot)  = (0.0, 0.0)
        else:
            # To test part A only, you can return None and the node will exit.
            return None

            # Part C (t>2):
            (alpha, alphadot) = ...  FIXME
            (beta,  betadot)  = ...  FIXME

        # Compute the desired rotation and angular velocity.
        FIXME: WHAT ARE Rd and wd?

        # Grab the last joint value and desired orientation.
        qdlast = self.qd
        Rdlast = self.Rd

        # Compute the inverse kinematics
        FIXME: INVERSE KINEMATICS FOR ROTATION

        # Integrate the joint position.
        qd = qdlast + dt * qddot

        # Save the joint value and desired orientation for next cycle.
        self.qd = qd
        self.Rd = Rd

        # Return the desired joint and task (orientation) pos/vel.
        return (qd, qddot, None, None, Rd, wd)


#
#  Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Initialize the generator node for 100Hz udpates, using the above
    # Trajectory class.
    generator = GeneratorNode('generator', 100, Trajectory)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the trajectory ends.
    generator.spin()

    # Shutdown the node and ROS.
    generator.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
