'''hw4p1sol.py

   This is the solution code for HW4 Problem 1.

   This simply uses NumPy to implement the known forward
   kinematics and Jacobian functions for the 3 DOF robot.

'''

import numpy as np


#
#  Link Lengths
#
l1 = 1
l2 = 1


#
#  Forward Kinematics
#
def fkin(q):
    # Precompute the sin/cos values from the joint vector.
    sp  = np.sin(q[0])
    cp  = np.cos(q[0])
    s1  = np.sin(q[1])
    c1  = np.cos(q[1])
    s12 = np.sin(q[1] + q[2])
    c12 = np.cos(q[1] + q[2])
    
    # Calculate the tip position.
    x = np.array([-sp * (l1 * c1 + l2 * c12),
                   cp * (l1 * c1 + l2 * c12),
                        (l1 * s1 + l2 * s12)])

    # Return the tip position as a NumPy 3-element vector.
    return x


#
#  Jacobian
#
def Jac(q):
    # Precompute the sin/cos values the 2D joint vector.
    sp  = np.sin(q[0])
    cp  = np.cos(q[0])
    s1  = np.sin(q[1])
    c1  = np.cos(q[1])
    s12 = np.sin(q[1] + q[2])
    c12 = np.cos(q[1] + q[2])
    
    # Calculate the tip position.
    J = np.array([[-cp * (l1*c1+l2*c12),  sp * (l1*s1+l2*s12),  sp * l2*s12],
                  [-sp * (l1*c1+l2*c12), -cp * (l1*s1+l2*s12), -cp * l2*s12],
                  [    0               ,       (l1*c1+l2*c12),       l2*c12]])

    # Return the Jacobian as a NumPy 3x3 matrix.
    return J


#
#  Main Code
#
#  This simply tests the above functions.
#
def main():
    # Run the test case.  Suppress infinitesimal numbers.
    np.set_printoptions(suppress=True)

    # First (given) test case with following joint coordinates.
    print("TEST CASE #1:")
    q = np.radians(np.array([20, 40, -30]))
    print('q:\n',       q)
    print('fkin(q):\n', fkin(q))
    print('Jac(q):\n',  Jac(q))

    # Second test case with following joint coordinates.
    print("TEST CASE #2")
    q = np.radians(np.array([30, 20, 50]))
    print('q:\n',       q)
    print('fkin(q):\n', fkin(q))
    print('Jac(q):\n',  Jac(q))

    # Third test case with following joint coordinates.
    print("TEST CASE #3")
    q = np.radians(np.array([0, 60, -120]))
    print('q:\n',       q)
    print('fkin(q):\n', fkin(q))
    print('Jac(q):\n',  Jac(q))

if __name__ == "__main__":
    main()
