'''hw5p4.py

   This is skeleton code for HW5 Problem 4.  Please EDIT.

   This moves the tip in a straight line (tip spline), then returns in
   a joint spline.

'''

import rclpy
import numpy as np

from math                       import pi, sin, cos, acos, atan2, sqrt, fmod

# Grab the utilities
from hw5code.TrajectoryUtils    import goto, spline, goto5, spline5
from hw5code.GeneratorNode      import GeneratorNode

# Also import the kinematics.
from hw4code.hw4p1              import fkin, Jac


#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node):
        # Define the known tip/joint positions.
        self.qA = np.radians(np.array([ 0, 60, -120]))
        self.xA = fkin(self.qA)

        self.qF = None
        self.xF = np.array([0.5, -0.5, 1.0])

        # Select the leg duration.
        self.T = 3.0

        # Initialize the parameters and anything stored between cycles!
        ....

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
        # End after one cycle.
        # if (t > 2*self.T):
        #     return None
        
        # First modulo the time by 2 legs.
        t = fmod(t, 2*self.T)

        # COMPUTE THE MOTION.
        ....

        # Return the desired joint and tip position and velocity.
        return (qd, qddot, xd, vd)


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
