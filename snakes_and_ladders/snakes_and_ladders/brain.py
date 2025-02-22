import cv2
import numpy as np
from math import sin, cos, pi, dist

import rclpy
import cv_bridge

from rclpy.node import Node

from geometry_msgs.msg import Point
from project_msgs.msg import Object, ObjectArray, PointArray, Segment, SegmentArray, State

from hw6sols.KinematicChainSol import KinematicChain
from snakes_and_ladders.constants import CYCLE, JOINT_NAMES, LOGGER_LEVEL, WAITING_POS


class DemoNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().set_level(LOGGER_LEVEL)
        self.get_logger().info('Name: %s' % name)

        self.chain = KinematicChain(self, 'world', 'tip', JOINT_NAMES[0:4])

        self.obj_arr_msg = ObjectArray()
        self.obj_arr_msg.objects = []
        self.point_array = []
        self.seg_arr_msg = SegmentArray()
        self.seg_arr_msg.segments = []
        self.x_waiting = []
        self.actual_pos = []
        self.bridge = cv_bridge.CvBridge()

        self.pub_segs = self.create_publisher(SegmentArray, name + '/segment_array', 1)
        
        self.sub_obj_array = self.create_subscription(
              ObjectArray, '/board_detector/object_array', self.recv_obj_array, 1)
        self.sub_state = self.create_subscription(
            State, '/trajectory/state', self.recv_state, 1)
        
        self.get_logger().info('Brain running...')


    def shutdown(self):
        self.destroy_node()


    def newton_raphson(self, x_goal):
        x_distance = []
        q_step_size = []
        q = self.actual_pos[0:4]  # exclude gripper
        N = 500

        for i in range(N+1):
            (x, _, Jv, _) = self.chain.fkin(q)
            x_delta = (x_goal - x)
            J = np.vstack([Jv, np.array([0, 1, -1, 1]).reshape(1,4)])
            x_stack = np.vstack([np.array(x_delta).reshape(3,1), np.array([0]).reshape(1,1)])
            q_delta = np.linalg.inv(J) @ x_stack
            q = q + q_delta.flatten() * 0.5
            x_distance.append(np.linalg.norm(x_delta))
            q_step_size.append(np.linalg.norm(q_delta))

            if np.linalg.norm(x-x_goal) < 1e-12:
                #self.get_logger().info("Completed in %d iterations" % i)
                return q.tolist()
            
        return WAITING_POS[0:4]
        

    def recv_state(self, msg):
        self.x_waiting = [msg.x_waiting_x, msg.x_waiting_y, msg.x_waiting_z]
        self.actual_pos = msg.actual_pos


    def recv_obj_array(self, msg):
        self.obj_arr_msg.objects = []

        for obj in msg.objects:
            self.obj_arr_msg.objects.append(obj)

        for obj in self.obj_arr_msg.objects:
            if obj.type == Object.DISK:
                disc_world_msg = Point()
                disc_world_msg.x = obj.x
                disc_world_msg.y = obj.y
                disc_world_msg.z = 0.012
                self.point_array.append(disc_world_msg)

        #self.get_logger().info('Objects: %s' % len(self.point_array))

        if len(self.point_array) > 0 and self.x_waiting != []:
            cart_points = [self.x_waiting]
            for pt in self.point_array:
                cart_points.append([pt.x, pt.y, pt.z])
                cart_points.append([pt.x + 0.06, pt.y, pt.z]) # move piece to next square
            self.point_array = []

            Tmove = CYCLE / 2

            for i in range(len(cart_points) - 1):
                #self.get_logger().info('Cart points: %s' % cart_points)
                p1 = cart_points[i]
                p2 = cart_points[i + 1]

                transitional = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, 0.09]  # xyz

                qT = self.newton_raphson(transitional)
                qT.append(0.0)  # gripper
                q2 = self.newton_raphson(p2)
                q2.append(0.0)  # gripper

                if i == 0:
                    segment = Segment()
                    segment.p = q2
                    segment.v = [0.0 for _ in segment.p]
                    segment.t = Tmove * 2
                    self.seg_arr_msg.segments.append(segment)

                    grip_segment = Segment()
                    grip_segment.p = q2
                    grip_segment.p[4] = -0.32
                    grip_segment.v = [0.0 for _ in grip_segment.p]
                    grip_segment.t = Tmove
                    self.seg_arr_msg.segments.append(grip_segment)

                    # release_segment = Segment()
                    # release_segment.p = q2
                    # release_segment.v = [0.0 for _ in release_segment.p]
                    # release_segment.t = Tmove
                    # self.seg_arr_msg.segments.append(release_segment)
                    continue

                dx = (transitional[0] - p1[0])
                dy = (transitional[1] - p1[1])
                v_cart = np.array([dx / Tmove, dy / Tmove, 0.0])

                (_, _, Jv, _) = self.chain.fkin(qT[0:4])
                J = np.vstack([Jv, np.array([0, 1, -1, 1]).reshape(1,4)])
                v_cart_stack = np.vstack([np.array(v_cart).reshape(3,1), np.array([0]).reshape(1,1)])
                qdotT = np.linalg.pinv(J) @ v_cart_stack
                qdotT = qdotT.flatten().tolist()
                qdotT.append(0.0)  # gripper

                seg1 = Segment()
                seg1.p = qT
                seg1.p[4] = -0.32 # gripper
                seg1.v = qdotT
                seg1.t = Tmove
                
                seg2 = Segment()
                seg2.p = q2
                seg2.p[4] = -0.32 # gripper
                seg2.v = [0.0 for _ in seg2.p]
                seg2.t = Tmove

                seg3 = Segment()
                seg3.p = q2
                seg3.v = [0.0 for _ in seg3.p]
                seg3.t = Tmove

                self.seg_arr_msg.segments.append(seg1)
                self.seg_arr_msg.segments.append(seg2)
                self.seg_arr_msg.segments.append(seg3)

            segment = Segment()
            segment.p = WAITING_POS
            segment.v = [0.0 for _ in segment.p]
            segment.t = Tmove * 2
            self.seg_arr_msg.segments.append(segment)


            self.pub_segs.publish(self.seg_arr_msg)
            self.get_logger().info('All segs: %s' % self.seg_arr_msg.segments)

            self.seg_arr_msg.segments = []
        else:
            self.get_logger().debug('this sucks')


def main(args=None):
    rclpy.init(args=args)
    node = DemoNode('brain')
    rclpy.spin(node)
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()