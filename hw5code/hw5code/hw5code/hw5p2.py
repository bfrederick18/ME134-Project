'''hw5p2.py

   This is skeleton code for HW5 Problem 2.  Please EDIT.

   Repeatedly and smoothly move the 3DOF.

'''

import rclpy
import numpy as np

from math                       import pi, sin, cos, acos, atan2, sqrt, fmod

# Grab the utilities
from hw5code.TrajectoryUtils    import goto, spline, goto5, spline5
from hw5code.GeneratorNode      import GeneratorNode


#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node):
        # Define the three joint positions.
        self.qA = np.radians(np.array([   0,  60, -120]))
        self.qB = np.radians(np.array([ -90, 135,  -90]))
        self.qC = np.radians(np.array([-180,  60, -120]))

    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names
        return ['theta1', 'theta2', 'theta3']

    # Evaluate the trajectory at the given time.  Note this may return:
    #   None                                The entire node stops
    #   (q, qdot)                           The joint position, velocity
    #   (q, qdot, p, v)                     The joint and task translation
    #   (q, qdot, p, v, R, omega)           The joint and task full pose
    #   (None, None, p, v)                  Just the task translation
    #   (None, None, None, None, R, omega)  Just the task orientation
    def evaluate(self, t, dt):
        # First modulo the time by 4 seconds
        t = fmod(t, 4.0)

        # Compute the joint values.
        if   (t < 2.0): (qd, qddot) = goto(t    , 2.0, self.qA, self.qB)
        else:           (qd, qddot) = goto(t-2.0, 2.0, self.qB, self.qA)

        # Return only the joint position and velocity.
        return (qd, qddot)


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
