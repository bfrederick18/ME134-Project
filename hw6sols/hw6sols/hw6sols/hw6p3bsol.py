'''hw6p3bsol.py

   This HW6 Problem 3, testing the kinematic chain.

   Please edit the KinematicChain code there!

'''

import rclpy
import numpy as np

from rclpy.node                 import Node

# Grab the kinematic chain
from hw6sols.KinematicChainSol  import KinematicChain


#
#  Define the test conditions
#
URDFNAME = 'kintester'

JOINTNAMES = ['theta1', 'theta2', 'd3', 'theta4']

TESTCASES = [
    (np.array( [ 0.123400, -0.427800, -0.509900, -0.248300]),
     np.array( [-1.293868,  0.034230,  0.612139]),
     np.array([[-0.277038, -0.208846,  0.937888],
               [-0.038704,  0.977726,  0.206285],
               [-0.960079,  0.020848, -0.278951]]),
     np.array([[ 0.028204,  0.001183, -0.437338,  0.062135],
               [-1.367153,  0.300169,  0.465227,  0.022994],
               [ 0.368863,  0.492096,  0.769610,  0.143564]]),
     np.array([[-0.022624,  0.342470,  0.      ,  0.874476],
               [ 0.259987,  0.801720,  0.      ,  0.246247],
               [ 0.965347, -0.489856,  0.      , -0.417916]])),

    (np.array( [-0.482400,  0.378500,  0.226800, -0.571800]),
     np.array( [-0.742813,  0.962192,  1.504422]),
     np.array([[-0.482976,  0.541096,  0.68844 ],
               [ 0.471970,  0.823102, -0.315826],
               [-0.737549,  0.172387, -0.65292 ]]),
     np.array([[-0.635619,  0.656591,  0.235074,  0.097891],
               [-0.815007, -0.731000,  0.076402, -0.070565],
               [ 0.204601,  0.471087,  0.968970,  0.102167]]),
     np.array([[-0.022624,  0.795837,  0.      ,  0.634423],
               [ 0.259987,  0.464517,  0.      , -0.200683],
               [ 0.965347, -0.388416,  0.      , -0.746481]])),
    ]


#
#  Run the test code
#
def main(args=None):
    # Set the print options to something we can read.
    np.set_printoptions(precision=3, suppress=True)

    # Initialize ROS and the node.
    rclpy.init(args=args)
    node = Node('kintest')

    # Report.
    print("Testing the kinematic chain on the %s" % URDFNAME)

    # Set up the kinematic chain object, assuming the 3 DOF.
    jointnames = JOINTNAMES
    baseframe  = 'world'
    tipframe   = 'tip'
    chain = KinematicChain(node, baseframe, tipframe, jointnames)

    # Define the check utility
    def check(name, answer, correct):
        print("%s(q)\n" % name, answer)
        fail = any(abs(answer - correct).flatten() > 1e-3)
        if fail:
            print("WARNING DIFFERENT FROM CORRECT %s:\n" % name, correct)
        return fail

    # Run the test cases.
    for case in TESTCASES:
        # Grab the case data:
        (q, ptip1, Rtip1, Jv1, Jw1) = case

        # Run the forward kinematics.
        (ptip, Rtip, Jv, Jw) = chain.fkin(q)

        # Print and compare:
        print('q:\n', q)
        fail = check('ptip', ptip, ptip1)
        fail = check('Rtip', Rtip, Rtip1) or fail
        fail = check('Jv', Jv, Jv1)       or fail
        fail = check('Jw', Jw, Jw1)       or fail
        print('----------------------------------------');

        # Abort if this failed.
        if (fail):
            raise Exception("Above Test Condition failed!")

    # Report success
    print("All above cases succeeded!")

    # Shutdown the node and ROS.
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
