#!/usr/bin/env python3
#
#   touchtable.py
#
#   Demonstration node to interact with the HEBIs.
#
import numpy as np
import rclpy

from math import sin, cos, pi

from rclpy.node         import Node
from sensor_msgs.msg    import JointState
from geometry_msgs.msg  import Point

from hw5code.TrajectoryUtils import goto, spline, goto5, spline5
from hw6sols.KinematicChainSol import KinematicChain


#
#   Definitions
#
RATE = 100.0            # Hertz
CYCLE = pi * 2
WAITING_POS = [0.0, 0.0, -pi/2, 0.0]
joint_names = ["base", "shoulder", "elbow", "wrist"]
urdf = 'fourdof'


#
#   DEMO Node Class
#
class DemoNode(Node):
    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        self.chain = KinematicChain(self, 'world', 'tip', joint_names)

        # Create a temporary subscriber to grab the initial position.
        self.position0 = self.grabfbk()
        # self.get_logger().info("Initial positions: %r" % self.position0)

        # Create a message and publisher to send the joint commands.
        self.cmdmsg = JointState()
        self.cmdpub = self.create_publisher(JointState, '/joint_commands', 10)

        # Wait for a connection to happen.  This isn't necessary, but
        # means we don't start until the rest of the system is ready.
        # self.get_logger().info("Waiting for a /joint_commands subscriber...")
        while(not self.count_subscribers('/joint_commands')):
            pass

        # Create a subscriber to continually receive joint state messages.
        self.actpos = self.position0.copy()
        self.fbksub = self.create_subscription(
            JointState, '/joint_states', self.recvfbk, 10)

        # Create a timer to keep calculating/sending commands.
        rate           = RATE
        self.starttime = self.get_clock().now()
        self.timer     = self.create_timer(1/rate, self.update)
        #self.get_logger().info("Sending commands with dt of %f seconds (%fHz)" %
                               #(self.timer.timer_period_ns * 1e-9, rate))
        
        self.pointsub = self.create_subscription(
            Point, '/point', self.recvpoint, 10)
        self.pointcmd = [0.0, 0.0, 0.0]

    # Shutdown
    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.destroy_node()


    # Grab a single feedback - DO NOT CALL THIS REPEATEDLY!
    def grabfbk(self):
        # Create a temporary handler to grab the position.
        def cb(fbkmsg):
            self.grabpos   = list(fbkmsg.position)
            self.grabready = True

        # Temporarily subscribe to get just one message.
        sub = self.create_subscription(JointState, '/joint_states', cb, 1)
        self.grabready = False
        while not self.grabready:
            rclpy.spin_once(self)
        self.destroy_subscription(sub)

        # Return the values.
        return self.grabpos

    # Send a command.
    def sendcmd(self, pos, vel, eff = []):
        # Build up the message and publish.
        self.cmdmsg.header.stamp = self.get_clock().now().to_msg()
        self.cmdmsg.name         = ['base', 'shoulder', 'elbow', 'wrist']
        self.cmdmsg.position     = pos
        self.cmdmsg.velocity     = vel
        self.cmdmsg.effort       = eff
        self.cmdpub.publish(self.cmdmsg)


    ######################################################################
    # Handlers
    # Receive feedback - called repeatedly by incoming messages.
    def recvfbk(self, fbkmsg):
        # Save the actual position.
        self.actpos = fbkmsg.position
    
    def recvpoint(self, pointmsg):
        # Extract the data.
        x = pointmsg.x
        y = pointmsg.y
        z = pointmsg.z
        self.pointcmd = [x, y, z]
        
        # Report.
        self.get_logger().info("Running point %r, %r, %r, %r" % (x,y,z,0))
    
    def super_smart_goto(self, t, initial_pos, final_pos, cycle):
        (q, qdot) = goto5(t % cycle, cycle, np.array(initial_pos).reshape(4, 1), np.array(final_pos).reshape(4, 1))
        return q.flatten().tolist(), qdot.flatten().tolist()

    def newton_raphson(self, xgoal):
        xdistance = []
        qstepsize = []
        q = self.actpos
        N = 500

        for i in range(N+1):
            #self.get_logger().info("q: %d, %d, %d, %d " % (q[0], q[1], q[2], q[3]))
            #self.get_logger().info(f"{q}")
            (x, _, Jv, _) = self.chain.fkin(q)
            xdelta = (xgoal - x)
            J = np.vstack([Jv, np.array([0, 1, 1, 1]).reshape(1,4)])
            qdelta = np.linalg.inv(J) @ np.vstack([np.array(xdelta).reshape(3,1), np.array([0]).reshape(1,1)])
            self.get_logger().info(f"{q}")
            q = q + qdelta.flatten() * 0.5
            self.get_logger().info(f"{q}")
            xdistance.append(np.linalg.norm(xdelta))
            qstepsize.append(np.linalg.norm(qdelta))

            if np.linalg.norm(x-xgoal) < 1e-12:
                #self.get_logger().info("Completed in %d iterations" % i)
                return q.tolist()
            
        return WAITING_POS

    # Timer (100Hz) update.
    def update(self):
        # Grab the current time.
        now = self.get_clock().now()
        t   = (now - self.starttime).nanoseconds * 1e-9
        qgoal = self.newton_raphson([1.36, 0.34, 0.02])

        if t < CYCLE:
            qd, qddot = self.super_smart_goto(t, self.position0, WAITING_POS, CYCLE)
        elif t < 2*CYCLE:
            qd, qddot = self.super_smart_goto(t, WAITING_POS, qgoal, CYCLE)
        else:
            self.actpos = WAITING_POS

        # Compute the trajectory.
        tau   = [0.0, 0.0, 0.0, 0.0]

        # Send.
        self.sendcmd(qd, qddot, tau)


#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the DEMO node.
    node = DemoNode('demo')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
