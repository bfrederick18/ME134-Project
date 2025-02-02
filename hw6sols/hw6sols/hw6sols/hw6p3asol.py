'''hw6p3asol.py

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
URDFNAME = 'threeDOF'

JOINTNAMES = ['theta1', 'theta2', 'theta3']

TESTCASES = [
    (np.array( [ 0.349066,  0.698132, -0.523599]),
     np.array( [-0.598827,  1.645263,  0.816436]),
     np.array([[ 0.939693, -0.336824,  0.059391],
               [ 0.34202 ,  0.925417, -0.163176],
               [ 0.      ,  0.173648,  0.984808]]),
     np.array([[-1.645263,  0.279237,  0.059391],
               [-0.598827, -0.767199, -0.163176],
               [ 0.      ,  1.750852,  0.984808]]),
     np.array([[ 0.      ,  0.939693,  0.939693],
               [ 0.      ,  0.34202 ,  0.34202 ],
               [ 1.      ,  0.      ,  0.      ]])),

    (np.array( [ 0.523599,  0.523599,  1.047198]),
     np.array( [-0.433013,  0.75    ,  1.5     ]),
     np.array([[ 0.866025, -0.      ,  0.5     ],
               [ 0.5     ,  0.      , -0.866025],
               [ 0.      ,  1.      ,  0.      ]]),
     np.array([[-0.75    ,  0.75    ,  0.5     ],
               [-0.433013, -1.299038, -0.866025],
               [ 0.      ,  0.866025,  0.      ]]),
     np.array([[ 0.      ,  0.866025,  0.866025],
               [ 0.      ,  0.5     ,  0.5     ],
               [ 1.      ,  0.      ,  0.      ]])),

    (np.array( [-0.785398,  1.308997,  2.094395]),
     np.array( [-0.5     , -0.5     ,  0.707107]),
     np.array([[ 0.707107, -0.683013,  0.183013],
               [-0.707107, -0.683013,  0.183013],
               [ 0.      , -0.258819, -0.965926]]),
     np.array([[ 0.5     , -0.5     ,  0.183013],
               [-0.5     , -0.5     ,  0.183013],
               [ 0.      , -0.707107, -0.965926]]),
     np.array([[ 0.      ,  0.707107,  0.707107],
               [ 0.      , -0.707107, -0.707107],
               [ 1.      ,  0.      ,  0.      ]])),
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
