import cv2
import numpy as np
from math import sin, cos, pi, dist

import matplotlib.pyplot as plt

import rclpy
import cv_bridge

from rclpy.node import Node

from geometry_msgs.msg import Point
from project_msgs.msg import Object, ObjectArray, PointArray, Segment, SegmentArray, BoxArray, State

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
        self.box_arr_msg = BoxArray()
        self.box_arr_msg.box = []
        self.x_waiting = []
        self.actual_pos = []
        self.board_positions = {}
        self.bridge = cv_bridge.CvBridge()

        self.pub_segs = self.create_publisher(SegmentArray, name + '/segment_array', 1)
        self.board_location = self.create_subscription(
            BoxArray, '/board_detector/box_array', self.recv_box_array, 1)
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
            #self.get_logger().info('All segs: %s' % self.seg_arr_msg.segments)

            self.seg_arr_msg.segments = []
        else:
            self.get_logger().debug('this sucks')
            
            
    def recv_box_array(self, msg):
        self.box_arr_msg.box = []
        
        for box in msg.box:
            self.box_arr_msg.box.append(box)
            
        w = 0.508
        h = 0.514
        # Calculate cell width & height (assuming a 10x10 board)
        cell_width = w / 10
        cell_height = h / 10    
        x_mid = self.box_arr_msg.box[0]
        y_mid = self.box_arr_msg.box[1]

        self.get_logger().info('Board Positions: %s, %s' % (x_mid, y_mid))

        self.board_positions = {}
        for row in range(5):
            for col in range(5):
                x_pos = x_mid - (4 - col)*cell_width - cell_width/2
                y_pos = y_mid - (4 - row)*cell_height - cell_height/2
                if row == 0:
                    cell_number = col + 1
                elif row == 1:
                    cell_number = 20 - col
                elif row == 2:
                    cell_number = 21 + col
                elif row == 3:
                    cell_number = 40 - col
                elif row == 4:
                    cell_number = 41 + col
                self.board_positions[cell_number] = (x_pos, y_pos)
        
        for row in range(5):
            for col in range(5, 10):
                x_pos = x_mid + (col - 5) * cell_width + cell_width/2
                y_pos = y_mid - (4 - row) * cell_height - cell_height/2
                if row == 0:
                    cell_number = col + 1
                elif row == 1:
                    cell_number = 20 - col
                elif row == 2:
                    cell_number = 21 + col
                elif row == 3:
                    cell_number = 40 - col
                elif row == 4:
                    cell_number = 41 + col
                self.board_positions[cell_number] = (x_pos, y_pos)
        
        for row in range(5, 10):
            for col in range(5, 10):
                x_pos = x_mid + (col - 5) * cell_width + cell_width/2
                y_pos = y_mid + (row - 5) * cell_height + cell_height/2
                if row == 5:
                    cell_number = 60 - col 
                elif row == 6:
                    cell_number = 61 + col
                elif row == 7:
                    cell_number = 80 - col
                elif row == 8:
                    cell_number = 81 + col
                elif row == 9:
                    cell_number = 100 - col
                self.board_positions[cell_number] = (x_pos, y_pos)
        
        for row in range(5, 10):
            for col in range(5):
                x_pos = x_mid - (4 - col)*cell_width - cell_width/2 
                y_pos = y_mid + (row - 5) * cell_height + cell_height/2
                if row == 5:
                    cell_number = 60 - col 
                elif row == 6:
                    cell_number = 61 + col
                elif row == 7:
                    cell_number = 80 - col
                elif row == 8:
                    cell_number = 81 + col
                elif row == 9:
                    cell_number = 100 - col
                self.board_positions[cell_number] = (x_pos, y_pos)

        #self.get_logger().info('Board Positions: %s' % board_positions)

        # Define ladders manually (start → end)
        ladders = {
            8: 27, 21: 41, 32: 51, 54: 66, 70: 89,
            77: 98
        }

        # Define snakes manually (start → end)
        snakes = {
            15: 4, 29: 12, 46: 18, 68: 49, 79: 57,
            95: 74
        }

def main(args=None):
    rclpy.init(args=args)
    node = DemoNode('brain')
    rclpy.spin(node)
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()