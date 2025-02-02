import rclpy

from rclpy.node                 import Node
from geometry_msgs.msg          import Point


class DemoNode(Node):
    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Create a subscriber to receive point messages.
        self.pointsub = self.create_subscription(
            Point, '/point', self.recvpoint, 10)

        # Report.
        self.get_logger().info("Running %s" % name)

    # Shutdown
    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.destroy_node()


    # Receive a point message - called by incoming messages.
    def recvpoint(self, pointmsg):
        # Extract the data.
        x = pointmsg.x
        y = pointmsg.y
        z = pointmsg.z
        
        # Report.
        self.get_logger().info("Running point %r, %r, %r" % (x,y,z))


def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the DEMO node.
    node = DemoNode('receivepoint')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()