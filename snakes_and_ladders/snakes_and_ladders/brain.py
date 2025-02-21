import cv2
import numpy as np
from math import sin, cos, pi, dist

import rclpy
import cv_bridge

from rclpy.node import Node

from geometry_msgs.msg import Point
from project_msgs.msg import Object, ObjectArray, PointArray, Segment, SegmentArray, State

from hw6sols.KinematicChainSol import KinematicChain
from snakes_and_ladders.constants import CYCLE, JOINT_NAMES, WAITING_POS


class DemoNode(Node):
    def __init__(self, name):
        super().__init__(name)

        self.chain = KinematicChain(self, 'world', 'tip', JOINT_NAMES)

        self.object_array = ObjectArray()
        self.object_array.objects = []

        self.point_array = []

        self.segment_array = SegmentArray()
        self.segment_array.segments = []

        self.x_waiting = []
        self.actpos = []

        self.pub_segs = self.create_publisher(SegmentArray, name + '/segment_array', 1)
        
        self.get_logger().info('Name: %s' % name)

        self.bridge = cv_bridge.CvBridge()
        
        self.sub_obj_array = self.create_subscription(
              ObjectArray, '/board_detector/object_array', self.recv_obj_array, 1)
        
        self.sub_state = self.create_subscription(
            State, '/trajectory/state', self.recv_state, 1)
        
        self.get_logger().info("Brain running...")


    def shutdown(self):
        self.destroy_node()


    def newton_raphson(self, xgoal):
        xdistance = []
        qstepsize = []
        q = self.actpos
        N = 500

        for i in range(N+1):
            (x, _, Jv, _) = self.chain.fkin(q)
            xdelta = (xgoal - x)
            J = np.vstack([Jv, np.array([0, 1, -1, 1]).reshape(1,4)])
            x_stack = np.vstack([np.array(xdelta).reshape(3,1), np.array([0]).reshape(1,1)])
            qdelta = np.linalg.inv(J) @ x_stack
            q = q + qdelta.flatten() * 0.5
            xdistance.append(np.linalg.norm(xdelta))
            qstepsize.append(np.linalg.norm(qdelta))

            if np.linalg.norm(x-xgoal) < 1e-12:
                #self.get_logger().info("Completed in %d iterations" % i)
                return q.tolist()
            
        return WAITING_POS
        

    def recv_state(self, msg):
        self.x_waiting = [msg.x_waiting_x, msg.x_waiting_y, msg.x_waiting_z]
        self.actpos = [msg.actpos_x, msg.actpos_y, msg.actpos_z, msg.actpos_w]


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
                self.point_array.append(disc_world_msg)
            
        #     elif obj.type == Object.STRIP:
        #         strip_world_start_msg = Point()
        #         strip_world_end_msg = Point()

        #         TAP_FACTOR = 0.04
        #         strip_world_start_msg.x = obj.x - TAP_FACTOR * np.cos(np.radians(obj.theta))
        #         strip_world_start_msg.y = obj.y + TAP_FACTOR * np.sin(np.radians(obj.theta))
        #         strip_world_end_msg.x = obj.x + TAP_FACTOR * np.cos(np.radians(obj.theta))
        #         strip_world_end_msg.y = obj.y - TAP_FACTOR * np.sin(np.radians(obj.theta))

        #         self.point_array.append(strip_world_start_msg)
        #         self.point_array.append(strip_world_end_msg)

        # point1 = Point()
        # point1.x = 0.54
        # point1.y = 0.26
        # point1.z = 0.02
        # self.point_array.append(point1)

        # point2 = Point()
        # point2.x = 0.64
        # point2.y = 0.26
        # point2.z = 0.02
        # self.point_array.append(point2)

        self.get_logger().info('Objects: %s' % len(self.point_array))

        if len(self.point_array) > 0 and self.x_waiting != []:
        #     self.pub_points.publish(self.point_array)
        #     # self.get_logger().info('All points: %s' % self.point_array.points)

        #     self.point_array.points = []

            cart_points = [self.x_waiting]
            for pt in self.point_array:
                cart_points.append([pt.x, pt.y, pt.z])
            self.point_array = []

            Tmove = CYCLE / 2

            for i in range(len(cart_points) - 1):
                p1 = cart_points[i]
                p2 = cart_points[i + 1]

                transitional = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, 0.07]

                qT = self.newton_raphson(transitional)
                q2 = self.newton_raphson(p2)

                if i == 0:
                    a_seg = Segment()
                    a_seg.p = q2
                    a_seg.v = [0.0, 0.0, 0.0, 0.0]
                    a_seg.t = Tmove * 2
                    self.segment_array.segments.append(a_seg)
                    continue

                dx = (transitional[0] - p1[0])
                dy = (transitional[1] - p1[1])
                v_cart = np.array([dx / Tmove, dy / Tmove, 0.0])

                (_, _, Jv, _) = self.chain.fkin(qT)
                J = np.vstack([Jv, np.array([0, 1, -1, 1]).reshape(1,4)])
                v_cart_stack = np.vstack([np.array(v_cart).reshape(3,1), np.array([0]).reshape(1,1)])
                qdotT = np.linalg.pinv(J) @ v_cart_stack
                qdotT = qdotT.flatten().tolist()

                seg1 = Segment()
                seg1.p = qT
                seg1.v = qdotT
                seg1.t = Tmove
                
                seg2 = Segment()
                seg2.p = q2
                seg2.v = [0.0, 0.0, 0.0, 0.0]
                seg2.t = Tmove

                self.segment_array.segments.append(seg1)
                self.segment_array.segments.append(seg2)

            a_seg = Segment()
            a_seg.p = WAITING_POS
            a_seg.v = [0.0, 0.0, 0.0, 0.0]
            a_seg.t = Tmove * 2
            self.segment_array.segments.append(a_seg)

            self.pub_segs.publish(self.segment_array)
            self.get_logger().info('All segs: %s' % self.segment_array.segments)

            self.segment_array.segments = []
        else:
            self.get_logger().info('this sucks')


def main(args=None):
    rclpy.init(args=args)
    node = DemoNode('brain')
    rclpy.spin(node)
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()