#!/usr/bin/env python3
#
#   testsend.py
#
#   Testing node to send joint commands.
#
import numpy as np
import rclpy

from rclpy.node         import Node
from sensor_msgs.msg    import JointState


#
#   Definitions
#
RATE     = 100.0        # Hertz


#
#   TEST Node Class
#
class TestNode(Node):
    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Add a publisher to send the joint commands.
        self.pub = self.create_publisher(JointState, '/joint_commands', 10)

        self.cmdmsg  = JointState()

        # Create a timer to keep calculating/sending commands.
        rate       = RATE
        self.timer = self.create_timer(1/rate, self.update)
        self.get_logger().info("Running with dt of %f seconds (%fHz)" %
                               (self.timer.timer_period_ns * 1e-9, rate))

    # Shutdown
    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.destroy_node()


    # Upate - called by the timesr
    def update(self):
        # Grab the current time.
        self.cmdtime = self.get_clock().now()

        # Build up the message and publish.  The joint names are preset.
        self.cmdmsg.header.stamp = self.cmdtime.to_msg()
        self.cmdmsg.name         = ['one', 'two']
        self.cmdmsg.position     = [1.0, 2.0]
        self.cmdmsg.velocity     = []
        self.cmdmsg.effort       = [0.0, 0.0]
        self.pub.publish(self.cmdmsg)


#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the TEST node.
    node = TestNode('testsend')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
