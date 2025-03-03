import cv2
import numpy as np
from math import sin, cos, pi, dist

import matplotlib.pyplot as plt

import rclpy
import cv_bridge

from rclpy.node import Node

from geometry_msgs.msg import Point
from project_msgs.msg import Object, ObjectArray, PointArray, Segment, SegmentArray, BoxArray, State, Num

from hw6sols.KinematicChainSol import KinematicChain
from snakes_and_ladders.constants import CYCLE, JOINT_NAMES, LOGGER_LEVEL, WAITING_POS, GRIPPER_CLOSE_DICE, GRIPPER_CLOSE_PURPLE


def create_seg(p, v=None, t=CYCLE / 2, gripper_val=0.0):
    seg = Segment()
    seg.p = p
    seg.p[4] = gripper_val
    seg.v = v if v != None else [0.0 for _ in seg.p]
    seg.t = t
    return seg


class DemoNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().set_level(LOGGER_LEVEL)
        self.get_logger().info('Name: %s' % name)

        self.chain = KinematicChain(self, 'world', 'tip', JOINT_NAMES[0:4])

        self.waiting_msg = None
        self.dice_roll = None
        self.check_board = False
        self.obt_board_positions = False
        self.received_dice_roll = False
        self.obj_arr_msg = ObjectArray()
        self.obj_arr_msg.objects = []
        self.point_array = []
        self.seg_arr_msg = SegmentArray()
        self.seg_arr_msg.segments = []
        self.box_arr_msg = BoxArray()
        self.box_arr_msg.box = []
        self.dice_face_msg = BoxArray()
        self.dice_face_msg.box = []
        self.x_waiting = []
        self.actual_pos = []
        self.board_positions = {}
        self.bridge = cv_bridge.CvBridge()
        self.counter = 0
        self.num_pub_dice = 1
        self.num_pub_player = 1
        self.reset = False
        self.position = 0
        self.prev_position = 0
        self.snakes = {}
        self.ladders = {}
        self.down_snake = False
        self.up_ladders = False

        self.pub_segs = self.create_publisher(SegmentArray, name + '/segment_array', 1)
        self.board_location = self.create_subscription(
            BoxArray, '/board_detector/box_array', self.recv_box_array, 1)
        self.sub_obj_array = self.create_subscription(
              ObjectArray, '/board_detector/object_array', self.recv_obj_array, 1)
        self.sub_state = self.create_subscription(
            State, '/trajectory/state', self.recv_state, 1)
        self.sub_check = self.create_subscription(
            Num, '/trajectory/num', self.recv_check, 1)
        self.sub_dice_roll = self.create_subscription(
            Num, '/dice_detector/num', self.recv_dice_roll, 1)
        self.sub_dice_array = self.create_subscription(
            BoxArray, '/dice_detector/box_array', self.recv_dice_box_array, 1)
        
        self.get_logger().info('Brain running...')


    def shutdown(self):
        self.destroy_node()


    def newton_raphson(self, x_goal, J_dict_val='vertical'):
        J_dict = {
            'vertical': np.array([0, 1, -1, 1]).reshape(1,4),
            'dice_bowl': np.array([0, 1.5, 1, 1]).reshape(1,4),
            'horizontal': np.array([0, 2.5, 1, 1]).reshape(1,4)  # not quite :(  I DONT KNOW HOW TO TUNE PLZ PAYAL HELP
        }

        x_distance = []
        q_step_size = []
        q = self.actual_pos[0:4]  # exclude gripper
        N = 500

        for i in range(N+1):
            (x, _, Jv, _) = self.chain.fkin(q)
            x_delta = (x_goal - x)
            J = np.vstack([Jv, J_dict[J_dict_val]])
            x_stack = np.vstack([np.array(x_delta).reshape(3,1), np.array([0]).reshape(1,1)])
            q_delta = np.linalg.inv(J) @ x_stack
            q = q + q_delta.flatten() * 0.1
            x_distance.append(np.linalg.norm(x_delta))
            q_step_size.append(np.linalg.norm(q_delta))

            if np.linalg.norm(x - x_goal) < 1e-12:
                #self.get_logger().info("Completed in %d iterations" % i)
                q_list = q.tolist()
                q_list.append(0.0)
                return q_list
                # return q.tolist()
            
        # return WAITING_POS[0:4]
        return WAITING_POS
        

    def recv_state(self, msg):
        self.x_waiting = [msg.x_waiting_x, msg.x_waiting_y, msg.x_waiting_z]
        self.actual_pos = msg.actual_pos
    

    def recv_dice_roll(self,msg):
        if msg.num is not None:
            self.dice_roll = msg.num


    def recv_check(self, msg):
        self.waiting_msg = msg.num
        if self.waiting_msg == 1:
            self.check_board = True
            self.counter += 1
            if self.counter % 2 == 0:
                self.received_dice_roll = True
                self.num_pub_player = 0
            elif self.counter % 2 == 1:
                self.received_dice_roll = False
                self.num_pub_dice = 0
            self.get_logger().info('Counter: %s' % self.counter)
        elif self.waiting_msg == 2:
            self.check_board = True
            self.reset = True
        # else:
        #     self.check_board = False
            #self.received_dice_roll = False


    def recv_obj_array(self, msg):
        if self.reset == True and self.obt_board_positions == True:
            self.get_logger().debug('resetting player position')
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

            if len(self.point_array) > 0 and self.x_waiting != []:
                cart_points = [self.x_waiting]
                
                for pt in self.point_array:
                    cart_points.append([pt.x, pt.y, pt.z + 0.10]) # PIECE POSITION
                    cart_points.append([pt.x, pt.y, pt.z]) # PIECE POSITION
                    self.get_logger().info('Reset point: %s, %s, %s' % (self.board_positions[1][0], self.board_positions[1][1], pt.z))
                    if self.board_positions[1][1] < 0.25:
                        cart_points.append([0.497, 0.271, pt.z])
                    else:
                        cart_points.append([self.board_positions[1][0], self.board_positions[1][1], pt.z])
                self.point_array = [] 

                Tmove = CYCLE / 2
                
                initial_player_pos_raise = cart_points[1]
                initial_player_pos = cart_points[2]
                reset_player_pos = cart_points[3]

                q4 = self.newton_raphson(initial_player_pos_raise)
                q5 = self.newton_raphson(initial_player_pos)

                
                self.seg_arr_msg.segments.append(create_seg(q4, t=2 * Tmove))  # above player position
                self.seg_arr_msg.segments.append(create_seg(q5))  # moving to player position
                self.seg_arr_msg.segments.append(create_seg(q5, gripper_val=GRIPPER_CLOSE_PURPLE))  # gripping player position


                transitional = [(initial_player_pos[0] + reset_player_pos[0]) / 2, (initial_player_pos[1] 
                                                                                    + reset_player_pos[1]) / 2, 0.09] 
                qT = self.newton_raphson(transitional, J_dict_val='vertical')
                q6 = self.newton_raphson(reset_player_pos, J_dict_val='vertical')

                dx = (transitional[0] - initial_player_pos[0])
                dy = (transitional[1] - initial_player_pos[1])
                v_cart = np.array([dx / Tmove, dy / Tmove, 0.0])

                (_, _, Jv, _) = self.chain.fkin(qT[0:4])
                J = np.vstack([Jv, np.array([0, 1, -1, 1]).reshape(1,4)])
                v_cart_stack = np.vstack([np.array(v_cart).reshape(3,1), np.array([0]).reshape(1,1)])
                qdotT = np.linalg.pinv(J) @ v_cart_stack
                qdotT = qdotT.flatten().tolist()
                qdotT.append(0.0)  # gripper

                self.seg_arr_msg.segments.append(create_seg(qT, v=qdotT, t=Tmove, gripper_val=GRIPPER_CLOSE_PURPLE))  # moving player position
                self.seg_arr_msg.segments.append(create_seg(q6, t=Tmove, gripper_val=GRIPPER_CLOSE_PURPLE))  # placing player position

                #releasing player position
                release_player_segment = Segment()
                release_player_segment.p = q6
                release_player_segment.p[4] = 0.0
                release_player_segment.v = [0.0 for _ in release_player_segment.p]
                release_player_segment.t = Tmove
                self.seg_arr_msg.segments.append(release_player_segment)
                
                #waiting 
                waiting_segment = Segment()
                waiting_segment.p = WAITING_POS
                waiting_segment.p[4] = 0.0
                waiting_segment.v = [0.0 for _ in waiting_segment.p]
                waiting_segment.t = Tmove * 2
                self.seg_arr_msg.segments.append(waiting_segment)

                self.pub_segs.publish(self.seg_arr_msg)
                self.num_pub_player += 1
                #self.get_logger().info('All segs: %s' % self.seg_arr_msg.segments)

                self.seg_arr_msg.segments = []

                self.position = 1

                self.reset = False
            
        elif self.received_dice_roll == True and self.counter % 2 == 0 and self.num_pub_player == 0:
            self.get_logger().debug('going to player')
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
                    self.prev_position = self.position
                    #cart_points.append([pt.x, pt.y, pt.z + 0.10]) # PIECE POSITION
                    cart_points.append([self.board_positions[self.prev_position][0], 
                                        self.board_positions[self.prev_position][1], pt.z + 0.10])
                    #cart_points.append([pt.x, pt.y, pt.z]) # PIECE POSITION
                    cart_points.append([self.board_positions[self.prev_position][0], 
                                        self.board_positions[self.prev_position][1], pt.z])

                    if self.dice_roll is not None:
                        new_pos = self.position + self.dice_roll
                        
                        if new_pos in self.snakes:
                            upd_pos = self.snakes[new_pos]
                            self.down_snake = True
                        elif new_pos in self.ladders:
                            upd_pos = self.ladders[new_pos]
                            self.up_ladders = True
                        cart_points.append([self.board_positions[new_pos][0], self.board_positions[new_pos][1], pt.z])
                        self.position = new_pos
                        final_player_pos = cart_points[3]
                        if self.down_snake == True or self.up_ladders == True:
                            cart_points.append([self.board_positions[upd_pos][0], self.board_positions[upd_pos][1], pt.z])
                            snake_player_pos = cart_points[4] #or ladder player position
                            self.position = upd_pos 
                        self.get_logger().info('Position: %s' % self.position)
                self.point_array = [] 

                Tmove = CYCLE / 2
                
                initial_player_pos_raise = cart_points[1]
                initial_player_pos = cart_points[2]

                q4 = self.newton_raphson(initial_player_pos_raise)
                q5 = self.newton_raphson(initial_player_pos)
                
                self.seg_arr_msg.segments.append(create_seg(q4, t= 2*Tmove))  # above player position
                self.seg_arr_msg.segments.append(create_seg(q5, t = Tmove))  # moving to player position
                self.seg_arr_msg.segments.append(create_seg(q5, gripper_val=GRIPPER_CLOSE_PURPLE))  # gripping player position

                if self.dice_roll is not None:
                    transitional = [(initial_player_pos[0] + final_player_pos[0]) / 2, (initial_player_pos[1] 
                                                                                        + final_player_pos[1]) / 2, 0.09] 
                    qT = self.newton_raphson(transitional)
                    q6 = self.newton_raphson(final_player_pos)

                    dx = (transitional[0] - initial_player_pos[0])
                    dy = (transitional[1] - initial_player_pos[1])
                    v_cart = np.array([dx / Tmove, dy / Tmove, 0.0])

                    (_, _, Jv, _) = self.chain.fkin(qT[0:4])
                    J = np.vstack([Jv, np.array([0, 1, -1, 1]).reshape(1,4)])
                    v_cart_stack = np.vstack([np.array(v_cart).reshape(3,1), np.array([0]).reshape(1,1)])
                    qdotT = np.linalg.pinv(J) @ v_cart_stack
                    qdotT = qdotT.flatten().tolist()
                    qdotT.append(0.0)  # gripper

                    self.seg_arr_msg.segments.append(create_seg(qT, v=qdotT, t=Tmove, gripper_val=GRIPPER_CLOSE_PURPLE))  # moving player position
                    self.seg_arr_msg.segments.append(create_seg(q6, t=Tmove, gripper_val=GRIPPER_CLOSE_PURPLE))  # placing player position

                    #releasing player position
                    release_player_segment = Segment()
                    release_player_segment.p = q6
                    release_player_segment.p[4] = 0.0
                    release_player_segment.v = [0.0 for _ in release_player_segment.p]
                    release_player_segment.t = Tmove

                    if self.down_snake == False and self.up_ladders == False:
                        self.seg_arr_msg.segments.append(release_player_segment)

                    if self.down_snake == True or self.up_ladders == True:
                        transitional2 = [(final_player_pos[0] + snake_player_pos[0]) / 2, (final_player_pos[1] 
                                                                                        + snake_player_pos[1]) / 2, 0.09] 
                        qT2 = self.newton_raphson(transitional2)
                        q7 = self.newton_raphson(snake_player_pos)

                        dx = (transitional2[0] - final_player_pos[0])
                        dy = (transitional2[1] - final_player_pos[1])
                        v_cart = np.array([dx / Tmove, dy / Tmove, 0.0])

                        (_, _, Jv, _) = self.chain.fkin(qT2[0:4])
                        J = np.vstack([Jv, np.array([0, 1, -1, 1]).reshape(1,4)])
                        v_cart_stack = np.vstack([np.array(v_cart).reshape(3,1), np.array([0]).reshape(1,1)])
                        qdotT = np.linalg.pinv(J) @ v_cart_stack
                        qdotT = qdotT.flatten().tolist()
                        qdotT.append(0.0)  # gripper

                        self.seg_arr_msg.segments.append(create_seg(qT2, v=qdotT, t=Tmove, gripper_val=GRIPPER_CLOSE_PURPLE))  # moving player position
                        self.seg_arr_msg.segments.append(create_seg(q7, t=Tmove, gripper_val=GRIPPER_CLOSE_PURPLE))  # placing player position

                        #releasing player position
                        release_player_segment = Segment()
                        release_player_segment.p = q7
                        release_player_segment.p[4] = 0.0
                        release_player_segment.v = [0.0 for _ in release_player_segment.p]
                        release_player_segment.t = Tmove
                        self.seg_arr_msg.segments.append(release_player_segment)

                        self.down_snake = False
                        self.up_ladders = False
                
                #waiting 
                waiting_segment = Segment()
                waiting_segment.p = WAITING_POS
                waiting_segment.p[4] = 0.0
                waiting_segment.v = [0.0 for _ in waiting_segment.p]
                waiting_segment.t = Tmove * 2
                self.seg_arr_msg.segments.append(waiting_segment)

                self.pub_segs.publish(self.seg_arr_msg)
                self.num_pub_player += 1
                #self.get_logger().info('All segs: %s' % self.seg_arr_msg.segments)

                self.seg_arr_msg.segments = []
                #self.received_dice_roll = False
            else:
                self.get_logger().debug('this sucks')
            

    def recv_dice_box_array(self, msg):
        if self.received_dice_roll == False and self.counter % 2 == 1 and self.num_pub_dice == 0:
            self.get_logger().debug('Rolling dice')
            self.dice_face_msg.box = []
            for box in msg.box:
                self.dice_face_msg.box.append(box)
            
            Tmove = CYCLE / 2
            dice_rest_pos = [self.dice_face_msg.box[0] + 0.025, self.dice_face_msg.box[1] - 0.01, 0.04]
            #dice_rest_pos = [1.338, 0.301, 0.04]
            lifted_dice_pos = [self.dice_face_msg.box[0] + 0.025, self.dice_face_msg.box[1] - 0.01, 0.11]
            #lifted_dice_pos = [1.338, 0.301, 0.11]

            q_dice_grip = self.newton_raphson(dice_rest_pos, J_dict_val='dice_bowl')
            q_dice_drop = self.newton_raphson(lifted_dice_pos, J_dict_val='horizontal')
            #q_dice_drop[3] = -np.pi/2
            
            self.seg_arr_msg.segments.append(create_seg(q_dice_grip, t=2 * Tmove))  # going to dice
            self.seg_arr_msg.segments.append(create_seg(q_dice_grip, t=Tmove, gripper_val=GRIPPER_CLOSE_DICE))  # gripping the dice
            self.seg_arr_msg.segments.append(create_seg(q_dice_drop, t=Tmove, gripper_val=GRIPPER_CLOSE_DICE))  # lifting dice up
            self.seg_arr_msg.segments.append(create_seg(q_dice_drop, t=Tmove))  # dropping the dice
            self.seg_arr_msg.segments.append(create_seg(WAITING_POS, t=2 * Tmove))  # waiting 

            self.pub_segs.publish(self.seg_arr_msg)
            self.num_pub_dice += 1
            self.seg_arr_msg.segments = []

            #self.received_dice_roll = True
            #self.counter = 1


    def recv_box_array(self, msg):
        if self.check_board == True:
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
            angle = self.box_arr_msg.box[2]

            #self.get_logger().info('Board Positions: %s, %s, %s' % (x_mid, y_mid, angle))

            #self.board_positions = {}
            # for row in range(10):
            #     for col in range(10):
            #         if col < 5:
            #             x_pos = x_mid - (((4 - col) * cell_width - cell_width / 2))*cos(np.radians(angle))
            #         else:
            #             x_pos = x_mid + ((col - 5) * cell_width + cell_width / 2)*cos(np.radians(angle))

            #         if row < 5:
            #             y_pos = y_mid - ((4 - row) * cell_height - cell_height / 2)*sin(np.radians(angle))
            #         else:
            #             y_pos = y_mid + ((row - 5) * cell_height + cell_height / 2)*sin(np.radians(angle))

            #         if row % 2 == 0:
            #             cell_number = 20 * (row // 2) + 1 + col
            #         else:
            #             cell_number = 20 * (((row - 1) // 2) + 1) - col
            
            self.board_positions = {}
            for row in range(5):
                for col in range(5):
                    x_pos_og = (x_mid - ((4 - col)*cell_width + cell_width/2))
                    y_pos_og = (y_mid - ((4 - row)*cell_height + cell_height/2)) 
                    x_pos = (x_pos_og - x_mid)*np.cos(np.radians(angle)) + (y_pos_og - y_mid)*np.sin(np.radians(angle)) + x_mid
                    y_pos = -(x_pos_og - x_mid)*np.sin(np.radians(angle)) + (y_pos_og - y_mid)*np.cos(np.radians(angle)) + y_mid
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
                    x_pos_og = x_mid + (col - 5) * cell_width + cell_width/2
                    y_pos_og = y_mid - (4 - row) * cell_height - cell_height/2
                    x_pos = (x_pos_og - x_mid)*np.cos(np.radians(angle)) + (y_pos_og - y_mid)*np.sin(np.radians(angle)) + x_mid
                    y_pos = -(x_pos_og - x_mid)*np.sin(np.radians(angle)) + (y_pos_og - y_mid)*np.cos(np.radians(angle)) + y_mid
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
                    self.board_positions[cell_number] = (x_pos + 0.022, y_pos - 0.02)
            
            for row in range(5, 10):
                for col in range(5, 10):
                    x_pos_og = x_mid + (col - 5) * cell_width + cell_width/2
                    y_pos_og = y_mid + (row - 5) * cell_height + cell_height/2
                    x_pos = (x_pos_og - x_mid)*np.cos(np.radians(angle)) + (y_pos_og - y_mid)*np.sin(np.radians(angle)) + x_mid
                    y_pos = -(x_pos_og - x_mid)*np.sin(np.radians(angle)) + (y_pos_og - y_mid)*np.cos(np.radians(angle)) + y_mid
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
                    self.board_positions[cell_number] = (x_pos + 0.022, y_pos)
            
            for row in range(5, 10):
                for col in range(5):
                    #x_pos = x_mid - (((4 - col)*cell_width - cell_width/2)*np.cos(np.radians(angle)) + ((row - 5) * (cell_height + cell_height/2)*np.sin(np.radians(angle))))
                    #y_pos = y_mid + (((row - 5) * (cell_height + cell_height/2)*np.cos(np.radians(angle))) - ((4 - col)*cell_width - cell_width/2)*np.sin(np.radians(angle)))
                    x_pos_og = x_mid - ((4 - col)*cell_width + cell_width/2)
                    y_pos_og = y_mid + ((row - 5) *cell_height + cell_height/2)
                    x_pos = (x_pos_og - x_mid)*np.cos(np.radians(angle)) + (y_pos_og - y_mid)*np.sin(np.radians(angle)) + x_mid
                    y_pos = -(x_pos_og - x_mid)*np.sin(np.radians(angle)) + (y_pos_og - y_mid)*np.cos(np.radians(angle)) + y_mid
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

            #self.get_logger().info('Board Positions: %s' % self.board_positions)

            # Define ladders manually (start → end)
            self.ladders = {
                8: 27, 21: 41, 32: 51, 54: 66, 70: 89,
                77: 98
            }

            # Define snakes manually (start → end)
            self.snakes = {
                15: 4, 29: 12, 46: 18, 68: 49, 79: 57,
                95: 74
            }
            if len(self.board_positions) != 0:
                self.obt_board_positions = True
                self.check_board = False
        else:
            self.board_positions = self.board_positions

def main(args=None):
    rclpy.init(args=args)
    node = DemoNode('brain')
    rclpy.spin(node)
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()