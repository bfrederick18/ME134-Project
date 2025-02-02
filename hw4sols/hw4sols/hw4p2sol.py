'''hw4p2sol.py

   This is the solution code for HW4 Problem 2.

   Implement the Newton-Raphson for seven target points.

'''

import numpy as np
import matplotlib.pyplot as plt

# Grab the fkin and Jac from P1.
from hw4sols.hw4p1sol import fkin, Jac, l1, l2


#
#  Utilities
#
# Determine the number of 360 deg wrappings:
def wraps(q):
    return np.round(q / (2*np.pi))

# 3 DOF Multiplicities - return True of False!
def elbow_up(q):
    return np.sin(q[2]) < 0.0

def front_side(q):
    return l1 * np.cos(q[1]) + l2 * np.cos(q[1] + q[2]) > 0.0



#
#  Newton Raphson
#
def newton_raphson(xgoal):
    # Collect the distance to goal and change in q every step!
    xdistance = []
    qstepsize = []

    # Set the initial joint value guess.
    q = np.array([0.0, np.pi/2, -np.pi/2])

    # Number of steps to try.
    N = 100

    # Iterate
    for i in range(N+1):
        # Determine where you are
        x = fkin(q)
        J = Jac(q)

        # Compute the delta and adjust.
        xdelta = (xgoal - x)
        qdelta = np.linalg.inv(J) @ xdelta
        q = q + qdelta

        # Store.
        xdistance.append(np.linalg.norm(xdelta))
        qstepsize.append(np.linalg.norm(qdelta))

        # Check whether to break.
        if np.linalg.norm(x-xgoal) < 1e-12:
            break


    # Final numbers.
    print("Target xgoal = ",xgoal," found q = ",q," after ",i, " steps.",
          "  Elbow Up ", elbow_up(q), " Front Side ", front_side(q),
          "  Wraps: ", wraps(q))

    # Create a plot of x distances to goal and q step sizes, for N steps.
    N = 20
    xdistance = xdistance[:N+1]
    qstepsize = qstepsize[:N+1]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
    ax1.plot(range(len(xdistance)), xdistance)
    ax2.plot(range(len(qstepsize)), qstepsize)

    ax1.set_title(f'Convergence Data for {xgoal.T}')
    ax2.set_xlabel('Iteration')

    ax1.set_ylabel('Task Distance to Goal')
    ax1.set_ylim([0, max(xdistance)])
    ax1.set_xlim([0, N])
    ax1.set_xticks(range(N+1))
    ax1.grid()

    ax2.set_ylabel('Joint Step Size')
    ax2.set_ylim([0, max(qstepsize)])
    ax2.set_xlim([0, N])
    ax2.set_xticks(range(N+1))
    ax2.grid()

    plt.show()


#
#  Main Code
#
def main():
    # Run the test case.  Suppress infinitesimal numbers.
    np.set_printoptions(suppress=True)

    # Prcess each target (goal position).
    for xgoal in [np.array([0.5,  1.0, 0.5]), 
                  np.array([1.0,  0.5, 0.5]),
                  np.array([2.0,  0.5, 0.5]),
                  np.array([0.0, -1.0, 0.5]),
                  np.array([0.0, -0.6, 0.5]),
                  np.array([0.5, -1.0, 0.5]),
                  np.array([-1.0, 0.0, 0.5])]:
        newton_raphson(xgoal)

if __name__ == "__main__":
    main()
