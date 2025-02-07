import cv2
import numpy as np

import rclpy
import cv_bridge

from rclpy.node import Node

from geometry_msgs.msg import Point
from project_msgs.msg import Object, ObjectArray, PointArray


class DemoNode(Node):
    def __init__(self, name):
        super().__init__(name)

        self.object_array = ObjectArray()
        self.object_array.objects = []

        self.point_array = PointArray()
        self.point_array.points = []

        self.pub_points = self.create_publisher(PointArray, name + '/points_array', 1)
        
        self.get_logger().info('Name: %s' % name)

        self.bridge = cv_bridge.CvBridge()
        
        self.sub_obj_array = self.create_subscription(
            ObjectArray, '/detector/object_array', self.recv_obj_array, 1)
        
        self.get_logger().info("Brain running...")


    def shutdown(self):
        self.destroy_node()


    def recv_obj_array(self, msg):
        self.object_array.objects = []

        for obj in msg.objects:
            self.object_array.objects.append(obj)

        for obj in self.object_array.objects:
            if obj.type == Object.DISK:
                disc_world_msg = Point()
                disc_world_msg.x = obj.x
                disc_world_msg.y = obj.y
                disc_world_msg.z = 0.012
                self.point_array.points.append(disc_world_msg)
            
            elif obj.type == Object.STRIP:
                strip_world_start_msg = Point()
                strip_world_end_msg = Point()

                TAP_FACTOR = 0.04
                strip_world_start_msg.x = obj.x - TAP_FACTOR * np.cos(np.radians(obj.theta))
                strip_world_start_msg.y = obj.y + TAP_FACTOR * np.sin(np.radians(obj.theta))
                strip_world_end_msg.x = obj.x + TAP_FACTOR * np.cos(np.radians(obj.theta))
                strip_world_end_msg.y = obj.y - TAP_FACTOR * np.sin(np.radians(obj.theta))

                self.point_array.points.append(strip_world_start_msg)
                self.point_array.points.append(strip_world_end_msg)

        if len(self.point_array.points) > 0:
            self.pub_points.publish(self.point_array)
            # self.get_logger().info('All points: %s' % self.point_array.points)

            self.point_array.points = []


def main(args=None):
    rclpy.init(args=args)
    node = DemoNode('brain')
    rclpy.spin(node)
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()