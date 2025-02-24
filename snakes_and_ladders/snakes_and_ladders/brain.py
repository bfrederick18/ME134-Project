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
from snakes_and_ladders.constants import CYCLE, JOINT_NAMES, LOGGER_LEVEL, WAITING_POS, GRIPPER_CLOSE_DICE, GRIPPER_CLOSE_PURPLE


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
            #J = np.vstack([Jv, np.array([0, 1.5, 1, 1]).reshape(1,4)])
            x_stack = np.vstack([np.array(x_delta).reshape(3,1), np.array([0]).reshape(1,1)])
            q_delta = np.linalg.inv(J) @ x_stack
            q = q + q_delta.flatten() * 0.5
            x_distance.append(np.linalg.norm(x_delta))
            q_step_size.append(np.linalg.norm(q_delta))

            if np.linalg.norm(x - x_goal) < 1e-12:
                #self.get_logger().info("Completed in %d iterations" % i)
                return q.tolist()
            
        return WAITING_POS[0:4]
    
    def newton_raphson_dice(self, x_goal):
        x_distance = []
        q_step_size = []
        q = self.actual_pos[0:4]  # exclude gripper
        N = 500

        for i in range(N+1):
            (x, _, Jv, _) = self.chain.fkin(q)
            x_delta = (x_goal - x)
            J = np.vstack([Jv, np.array([0, 1.5, 1, 1]).reshape(1,4)])
            #J = np.vstack([Jv, np.array([0, 1, -1, 1]).reshape(1,4)])
            x_stack = np.vstack([np.array(x_delta).reshape(3,1), np.array([0]).reshape(1,1)])
            q_delta = np.linalg.inv(J) @ x_stack
            q = q + q_delta.flatten() * 0.5
            x_distance.append(np.linalg.norm(x_delta))
            q_step_size.append(np.linalg.norm(q_delta))

            if np.linalg.norm(x-x_goal) < 1e-12:
                #self.get_logger().info("Completed in %d iterations" % i)
                q_list = q.tolist()
                return q.tolist()
            
        return WAITING_POS[0:4]
        

    def recv_state(self, msg):
        self.x_waiting = [msg.x_waiting_x, msg.x_waiting_y, msg.x_waiting_z]
        self.actual_pos = msg.actual_pos


    def create_seg(self, p, v=None, t=CYCLE / 2, gripper_val=0.0):
        seg = Segment()
        seg.p = p
        seg.p[4] = gripper_val
        seg.v = v if v != None else [0.0 for _ in seg.p]
        seg.t = t
        return seg


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
            cart_points.append([1.32, 0.285, 0.04])  # DICE HARDCODE DELETE LATER
            cart_points.append([1.32, 0.285, 0.09])  # LIFTED DICE POSITION (HARDCODE)

            for pt in self.point_array:
                cart_points.append([pt.x, pt.y, pt.z]) # PIECE POSITION
                cart_points.append([pt.x + 0.06, pt.y, pt.z])  # MOVE PIECE TO NEXT SQUARE
            self.point_array = [] 

            Tmove = CYCLE / 2

            waiting_pos = cart_points[0]
            dice_rest_pos = cart_points[1]
            lifted_dice_pos = cart_points[2]
            initial_player_pos = cart_points[3]
            final_player_pos = cart_points[4]

            q2 = self.newton_raphson_dice(dice_rest_pos)
            q2.append(0.0)

            q3 = self.newton_raphson_dice(lifted_dice_pos)
            q3.append(0.0)
            
            q4 = self.newton_raphson_dice(initial_player_pos)
            q4.append(0.0)

            transitional = [(initial_player_pos[0] + final_player_pos[0]) / 2, (initial_player_pos[1] 
                                                                                + final_player_pos[1]) / 2, 0.09] 
            qT = self.newton_raphson_dice(transitional)
            qT.append(0.0)
            
            q5 = self.newton_raphson_dice(final_player_pos)
            q5.append(0.0)

            #going to dice
            segment1 = Segment()
            segment1.p = q2
            segment1.v = [0.0 for _ in segment1.p]
            segment1.t = Tmove 
            self.seg_arr_msg.segments.append(segment1)

            #gripping the dice
            grip_segment = Segment()
            grip_segment.p = q2
            grip_segment.p[4] = GRIPPER_CLOSE_DICE
            grip_segment.v = [0.0 for _ in grip_segment.p]
            grip_segment.t = Tmove
            self.seg_arr_msg.segments.append(grip_segment)

            #lifting dice up
            lift_segment = Segment()
            lift_segment.p = q3
            lift_segment.p[4] = GRIPPER_CLOSE_DICE
            lift_segment.v = [0.0 for _ in lift_segment.p]
            lift_segment.t = Tmove
            self.seg_arr_msg.segments.append(lift_segment)

            #dropping the dice
            drop_segment = Segment()
            drop_segment.p = q3
            drop_segment.p[4] = 0.0
            drop_segment.v = [0.0 for _ in drop_segment.p]
            drop_segment.t = Tmove
            self.seg_arr_msg.segments.append(drop_segment)

            #moving to player position
            player_segment = Segment()
            player_segment.p = q4
            player_segment.p[4] = 0.0
            player_segment.v = [0.0 for _ in player_segment.p]
            player_segment.t = 2*Tmove
            self.seg_arr_msg.segments.append(player_segment)

            #gripping player position
            lift_player_segment = Segment()
            lift_player_segment.p = q4
            lift_player_segment.p[4] = GRIPPER_CLOSE_PURPLE
            lift_player_segment.v = [0.0 for _ in lift_player_segment.p]
            lift_player_segment.t = Tmove
            self.seg_arr_msg.segments.append(lift_player_segment)

            dx = (transitional[0] - initial_player_pos[0])
            dy = (transitional[1] - initial_player_pos[1])
            v_cart = np.array([dx / Tmove, dy / Tmove, 0.0])

            (_, _, Jv, _) = self.chain.fkin(qT[0:4])
            J = np.vstack([Jv, np.array([0, 1, -1, 1]).reshape(1,4)])
            v_cart_stack = np.vstack([np.array(v_cart).reshape(3,1), np.array([0]).reshape(1,1)])
            qdotT = np.linalg.pinv(J) @ v_cart_stack
            qdotT = qdotT.flatten().tolist()
            qdotT.append(0.0)  # gripper


            #moving player position
            first_transition_segment = Segment()
            first_transition_segment.p = qT
            first_transition_segment.p[4] = GRIPPER_CLOSE_PURPLE
            first_transition_segment.v = qdotT 
            first_transition_segment.t = Tmove
            self.seg_arr_msg.segments.append(first_transition_segment)

            #placing player position
            second_transition_segment = Segment()
            second_transition_segment.p = q5
            second_transition_segment.p[4] = GRIPPER_CLOSE_PURPLE
            second_transition_segment.v = [0.0 for _ in drop_segment.p]
            second_transition_segment.t = Tmove
            self.seg_arr_msg.segments.append(second_transition_segment)

            #releasing player position
            release_player_segment = Segment()
            release_player_segment.p = q5
            release_player_segment.p[4] = 0.0
            release_player_segment.v = [0.0 for _ in drop_segment.p]
            release_player_segment.t = Tmove
            self.seg_arr_msg.segments.append(release_player_segment)
            
            #waiting 
            waiting_segment = Segment()
            waiting_segment.p = WAITING_POS
            waiting_segment.p[4] = 0.0
            waiting_segment.v = [0.0 for _ in waiting_segment.p]
            waiting_segment.t = Tmove * 2
            self.seg_arr_msg.segments.append(waiting_segment)

            #region
            """
            #self.get_logger().info('Cart points: %s' % cart_points)
            p1 = cart_points[0]
            p2 = cart_points[1]

            transitional = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, 0.09]  # xyz

            qT = self.newton_raphson_dice(transitional)
            qT.append(0.0)  # gripper
            q2 = self.newton_raphson_dice(p2)
            q2.append(0.0)  # gripper

            p3 = cart_points[2]
            q2 = self.newton_raphson_dice(p2)
            q3 = self.newton_raphson_dice(p3)

            q2.append(0.0)  # gripper
            q3.append(0.0)
            #going to dice
            segment = Segment()
            segment.p = q2
            segment.v = [0.0 for _ in segment.p]
            segment.t = Tmove * 2
            self.seg_arr_msg.segments.append(segment)
            #gripping the dice
            grip_segment = Segment()
            grip_segment.p = q2
            grip_segment.p[4] = GRIPPER_CLOSE_DICE
            grip_segment.v = [0.0 for _ in grip_segment.p]
            grip_segment.t = Tmove
            self.seg_arr_msg.segments.append(grip_segment)
            #lifting dice up
            lift_segment = Segment()
            lift_segment.p = q3
            lift_segment.p[4] = GRIPPER_CLOSE_DICE
            lift_segment.v = [0.0 for _ in lift_segment.p]
            lift_segment.t = Tmove
            self.seg_arr_msg.segments.append(lift_segment)
            #dropping the dice
            drop_segment = Segment()
            drop_segment.p = q3
            drop_segment.p[4] = 0.0
            drop_segment.v = [0.0 for _ in drop_segment.p]
            drop_segment.t = Tmove
            self.seg_arr_msg.segments.append(drop_segment)

            segment = Segment()
            segment.p = WAITING_POS
            segment.v = [0.0 for _ in segment.p]
            segment.t = Tmove * 2
            self.seg_arr_msg.segments.append(segment)

            dx = (transitional[0] - p1[0])
            dy = (transitional[1] - p1[1])
            v_cart = np.array([dx / Tmove, dy / Tmove, 0.0])

            (_, _, Jv, _) = self.chain.fkin(qT[0:4])
            J = np.vstack([Jv, np.array([0, 1, -1, 1]).reshape(1,4)])
            v_cart_stack = np.vstack([np.array(v_cart).reshape(3,1), np.array([0]).reshape(1,1)])
            qdotT = np.linalg.pinv(J) @ v_cart_stack
            qdotT = qdotT.flatten().tolist()
            qdotT.append(0.0)  # gripper

            release_segment = Segment()
            release_segment.p = q2
            release_segment.v = [0.0 for _ in release_segment.p]
            release_segment.t = Tmove
            self.seg_arr_msg.segments.append(release_segment)

            seg1 = Segment()
            seg1.p = q2
            seg1.p[4] = GRIPPER_CLOSE_PURPLE
            seg1.v = [0.0 for _ in seg2.p]
            seg1.t = 2*Tmove

            seg2 = Segment()
            seg2.p = qT
            seg2.p[4] = GRIPPER_CLOSE_PURPLE
            seg2.v = qdotT
            seg2.t = Tmove
            
            seg3 = Segment()
            seg3.p = q2
            seg3.p[4] = GRIPPER_CLOSE_PURPLE
            seg3.v = [0.0 for _ in seg2.p]
            seg3.t = 2*Tmove

            seg4 = Segment()
            seg4.p = q2
            seg4.v = [0.0 for _ in seg3.p]
            seg4.t = Tmove

            self.seg_arr_msg.segments.append(seg1)
            self.seg_arr_msg.segments.append(seg2)
            self.seg_arr_msg.segments.append(seg3)
            self.seg_arr_msg.segments.append(seg4)

            segment = Segment()
            segment.p = WAITING_POS
            segment.v = [0.0 for _ in segment.p]
            segment.t = Tmove * 2
            self.seg_arr_msg.segments.append(segment)
            """
            #endregion

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
        for row in range(10):
            for col in range(10):
                if col < 5:
                    x_pos = x_mid - (4 - col) * cell_width - cell_width / 2
                else:
                    x_pos = x_mid + (col - 5) * cell_width + cell_width / 2

                if row < 5:
                    y_pos = y_mid - (4 - row) * cell_height - cell_height / 2
                else:
                    y_pos = y_mid + (row - 5) * cell_height + cell_height / 2

                if row % 2 == 0:
                    cell_number = 20 * (row // 2) + 1 + col
                else:
                    cell_number = 20 * (((row - 1) // 2) + 1) - col

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