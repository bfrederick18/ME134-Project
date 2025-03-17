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
from snakes_and_ladders.constants import CYCLE, JOINT_NAMES, LOGGER_LEVEL, WAITING_POS, GRIPPER_CLOSE_DICE, GRIPPER_CLOSE_PURPLE, GRIPPER_INTERMEDIATE


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
        self.blue_obj_arr_msg = ObjectArray()
        self.blue_obj_arr_msg.objects = []
        self.point_array = []
        self.seg_arr_msg = SegmentArray()
        self.seg_arr_msg.segments = []
        self.box_arr_msg = BoxArray()
        self.box_arr_msg.box = []
        self.dice_face_msg = BoxArray()
        self.dice_face_msg.box = []
        self.x_waiting = []
        self.actual_pos = []
        self.actual_vel = []
        self.actual_eff = []
        self.board_positions = {}
        self.bridge = cv_bridge.CvBridge()
        self.counter = 0
        self.blue_counter = 0
        self.num_pub_dice = 1
        self.num_pub_player = 1
        self.reset = False
        self.position = 0
        self.prev_position = 0
        self.snakes = {}
        self.ladders = {}
        self.down_snake = False
        self.up_ladders = False
        self.reset_pt = False
        self.curr_player_pos = []
        self.prev_player_pos = []
        self.human_player_pos = 0
        self.prev_human_player_pos = 0
        self.dice_hidden = False
        self.dice_hidden_counter = 0
        self.recover_purple = 0
        self.purple_disk_coords = []

        self.pub_segs = self.create_publisher(SegmentArray, name + '/segment_array', 1)
        self.board_location = self.create_subscription(
            BoxArray, '/board_detector/box_array', self.recv_box_array, 1)
        self.sub_obj_array = self.create_subscription(
              ObjectArray, '/board_detector/object_array', self.recv_obj_array, 1)
        self.sub_blue_obj_array = self.create_subscription(
              ObjectArray, '/board_detector/blue_object_array', self.recv_blue_obj_array, 1)
        self.sub_state = self.create_subscription(
            State, '/trajectory/state', self.recv_state, 1)
        self.sub_check = self.create_subscription(
            Num, '/trajectory/num', self.recv_check, 10)
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
            'horizontal': np.array([0, 2.5, 1, 1]).reshape(1,4)
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
                q_list = q.tolist()
                q_list.append(0.0)
                return q_list
            
        self.get_logger().info("Newton-Raphson failed to converge in %d iterations" % N)
        return WAITING_POS


    def create_transitional(self, initial_segment, final_segment, Tmove):
        transitional = [(initial_segment[0] + final_segment[0]) / 2, (initial_segment[1] + final_segment[1]) / 2, 0.10] 
        qT = self.newton_raphson(transitional, J_dict_val='vertical')

        dx = (transitional[0] - initial_segment[0])
        dy = (transitional[1] - initial_segment[1])
        v_cart = np.array([dx / Tmove, dy / Tmove, 0.0])

        (_, _, Jv, _) = self.chain.fkin(qT[0:4])
        J = np.vstack([Jv, np.array([0, 1, -1, 1]).reshape(1,4)])
        v_cart_stack = np.vstack([np.array(v_cart).reshape(3,1), np.array([0]).reshape(1,1)])
        qdotT = np.linalg.pinv(J) @ v_cart_stack
        qdotT = qdotT.flatten().tolist()
        qdotT.append(0.0)  # gripper
        return qT, qdotT
    
    def create_transitional_modified(self, initial_segment, Tmove):
        transitional = [initial_segment[0], initial_segment[1], 0.09] 
        qT = self.newton_raphson(transitional, J_dict_val='vertical')

        dx = (transitional[0] - initial_segment[0])
        dy = (transitional[1] - initial_segment[1])
        dz = (transitional[2] - initial_segment[2])
        v_cart = np.array([dx / Tmove, dy / Tmove, dz / Tmove])

        (_, _, Jv, _) = self.chain.fkin(qT[0:4])
        J = np.vstack([Jv, np.array([0, 1, -1, 1]).reshape(1,4)])
        v_cart_stack = np.vstack([np.array(v_cart).reshape(3,1), np.array([0]).reshape(1,1)])
        qdotT = np.linalg.pinv(J) @ v_cart_stack
        qdotT = qdotT.flatten().tolist()
        qdotT.append(0.0)  # gripper
        return qT, qdotT
        

    def recv_state(self, msg):
        self.x_waiting = [msg.x_waiting_x, msg.x_waiting_y, msg.x_waiting_z]
        self.actual_pos = msg.actual_pos
        self.actual_vel = msg.actual_vel
        self.actual_eff = msg.actual_eff
        #self.get_logger().info('Effort: %s' % self.actual_eff)
        
    

    def recv_dice_roll(self,msg):
        if msg.num is not None:
            if msg.num == 0:
                self.dice_hidden_counter += 1
            if self.dice_hidden_counter == 15:
                self.dice_hidden = True
                self.dice_hidden_counter = 0
            elif msg.num != 0:
                self.dice_roll = msg.num

    def recv_check(self, msg):
        self.waiting_msg = msg.num

        if self.waiting_msg == 1:
            self.obt_board_positions = False
            self.check_board = True
            self.counter += 1
            self.get_logger().info('Counter: %s' % self.counter)
            if self.counter % 2 == 0:
                self.received_dice_roll = True
                self.num_pub_player = 0
            elif self.counter % 2 == 1:
                self.received_dice_roll = False
                self.num_pub_dice = 0

        elif self.waiting_msg == 2:
            self.check_board = True
            self.reset = True


    def recv_blue_obj_array(self, msg):
        for obj in msg.objects:
            if obj.type == Object.BLUE_DISK:
                player_world_msg = Point()
                player_world_msg.x = obj.x
                player_world_msg.y = obj.y
                player_world_msg.z = 0.012
                self.blue_counter += 1
            if self.blue_counter == 15:
                self.prev_player_pos = self.curr_player_pos
                self.curr_player_pos = [player_world_msg.x, player_world_msg.y, player_world_msg.z]
                self.blue_counter = 0


    def recv_obj_array(self, msg):
        self.obj_arr_msg.objects = []
        for obj in msg.objects:
            self.obj_arr_msg.objects.append(obj)

        for obj in self.obj_arr_msg.objects:
            if obj.type == Object.PURPLE_DISK:
                disc_world_msg = Point()
                disc_world_msg.x = obj.x
                disc_world_msg.y = obj.y
                disc_world_msg.z = 0.012
                self.purple_disk_coords = [obj.x, obj.y, 0.012]

        if self.reset == True and self.obt_board_positions == True:
            self.get_logger().debug('resetting player position')
            # self.obj_arr_msg.objects = []
            # for obj in msg.objects:
            #     self.obj_arr_msg.objects.append(obj)

            # for obj in self.obj_arr_msg.objects:
            #     if obj.type == Object.PURPLE_DISK:
            #         disc_world_msg = Point()
            #         disc_world_msg.x = obj.x
            #         disc_world_msg.y = obj.y
            #         disc_world_msg.z = 0.012
            #         self.point_array.append(disc_world_msg)
            #         self.purple_disk_coords = [obj.x, obj.y, 0.012]
            self.point_array.append(disc_world_msg)
            if len(self.point_array) > 0 and self.x_waiting != []:
                cart_points = [self.x_waiting]
                
                for pt in self.point_array:
                    cart_points.append([pt.x, pt.y, pt.z + 0.10]) # PIECE POSITION
                    cart_points.append([pt.x, pt.y, pt.z]) # PIECE POSITION
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

                self.seg_arr_msg.segments.append(create_seg(q4, t=2*Tmove))  # above player position
                self.seg_arr_msg.segments.append(create_seg(q5, gripper_val=GRIPPER_INTERMEDIATE))  # moving to player position
                self.seg_arr_msg.segments.append(create_seg(q5, gripper_val=GRIPPER_CLOSE_PURPLE))  # gripping player position

                qT, qdotT = self.create_transitional(initial_player_pos, reset_player_pos, Tmove)
                q6 = self.newton_raphson(reset_player_pos, J_dict_val='vertical')

                self.seg_arr_msg.segments.append(create_seg(qT, v=qdotT, t=Tmove, gripper_val=GRIPPER_CLOSE_PURPLE))  # moving player position
                self.seg_arr_msg.segments.append(create_seg(q6, t=Tmove, gripper_val=GRIPPER_CLOSE_PURPLE))  # placing player position
                self.seg_arr_msg.segments.append(create_seg(q6, t=Tmove))
                self.seg_arr_msg.segments.append(create_seg(WAITING_POS, t=2*Tmove))
                self.seg_arr_msg.segments.append(create_seg(WAITING_POS, t=pi/4, gripper_val=GRIPPER_CLOSE_DICE))
                self.seg_arr_msg.segments.append(create_seg(WAITING_POS, t=pi/4))

                self.pub_segs.publish(self.seg_arr_msg)
                self.num_pub_player += 1
            
                self.seg_arr_msg.segments = []

                self.position = 1

                self.reset = False
            
        elif self.received_dice_roll == True and self.counter % 2 == 0 and self.num_pub_player == 0:
            self.get_logger().debug('going to player')
            self.obj_arr_msg.objects = []

            for obj in msg.objects:
                self.obj_arr_msg.objects.append(obj)

            for obj in self.obj_arr_msg.objects:
                if obj.type == Object.PURPLE_DISK:
                    disc_world_msg = Point()
                    disc_world_msg.x = obj.x
                    disc_world_msg.y = obj.y
                    disc_world_msg.z = 0.012
                    self.point_array.append(disc_world_msg)

            if len(self.point_array) > 0 and self.x_waiting != []:
                cart_points = [self.x_waiting]
                
                for pt in self.point_array:
                    self.prev_position = self.position

                    if abs(self.board_positions[self.prev_position][0] - pt.x) >= 0.04 or abs(self.board_positions[self.prev_position][1] - pt.y) >= 0.04:
                        self.reset_pt = True
                        cart_points.append([pt.x, pt.y, pt.z + 0.10])
                        cart_points.append([pt.x, pt.y, pt.z])
                        cart_points.append([self.board_positions[self.prev_position][0], 
                                            self.board_positions[self.prev_position][1], pt.z])
                        reset_point_raise = cart_points[1]
                        reset_point = cart_points[2]
                        initial_player_pos = cart_points[3]
                        q4 = self.newton_raphson(reset_point_raise)
                        q5 = self.newton_raphson(reset_point)
                        q_real = self.newton_raphson(initial_player_pos)

                    else:
                        self.reset_pt = False
                        cart_points.append([pt.x, pt.y, pt.z + 0.10])
                        cart_points.append([pt.x, pt.y, pt.z])
                        initial_player_pos_raise = cart_points[1]
                        initial_player_pos = cart_points[2]
                        q4 = self.newton_raphson(initial_player_pos_raise)
                        q5 = self.newton_raphson(initial_player_pos)
                    
                    self.get_logger().info('Need to reset? %s' % self.reset_pt)

                    if self.dice_roll is not None:
                        if self.position == 100:
                            self.human_player_pos = 0
                            self.counter = 0
                            self.check_board = True
                            self.reset = True
                            return
                        else:
                            new_pos = self.position + self.dice_roll
                            if new_pos >= 100:
                                self.get_logger().debug('R2-D2 WINS!!!! :)')
                                new_pos = 100

                            if new_pos in self.snakes:
                                upd_pos = self.snakes[new_pos]
                                self.down_snake = True
                            elif new_pos in self.ladders:
                                upd_pos = self.ladders[new_pos]
                                self.up_ladders = True

                            cart_points.append([self.board_positions[new_pos][0], self.board_positions[new_pos][1], pt.z])
                            self.position = new_pos

                            if self.reset_pt == True:
                                final_player_pos = cart_points[4]
                            else:
                                final_player_pos = cart_points[3]

                            if self.down_snake == True or self.up_ladders == True:
                                cart_points.append([self.board_positions[upd_pos][0], self.board_positions[upd_pos][1], pt.z])
                                if self.reset_pt == True:
                                    snake_player_pos = cart_points[5] #or ladder player position
                                else:
                                    snake_player_pos = cart_points[4] #or ladder player position
                                self.position = upd_pos 

                        self.get_logger().info('Position: %s' % self.position)
                        
                self.point_array = [] 
                Tmove = CYCLE / 2

                self.seg_arr_msg.segments.append(create_seg(q4, t=2*Tmove))  # above player position
                self.seg_arr_msg.segments.append(create_seg(q5, t=Tmove, gripper_val=GRIPPER_INTERMEDIATE))  # moving to player position
                self.seg_arr_msg.segments.append(create_seg(q5, gripper_val=GRIPPER_CLOSE_PURPLE))  # gripping player position

                if self.reset_pt == True:
                    qTreal, qdotTreal = self.create_transitional(reset_point, initial_player_pos, Tmove)
                    self.seg_arr_msg.segments.append(create_seg(qTreal, v=qdotTreal, t=Tmove, gripper_val=GRIPPER_CLOSE_PURPLE))
                    self.seg_arr_msg.segments.append(create_seg(q_real, t=Tmove, gripper_val=GRIPPER_CLOSE_PURPLE))
                    self.reset_pt = False

                if self.dice_roll is not None:
                    #qT, qdotT = self.create_transitional(initial_player_pos, final_player_pos, Tmove)
                    qT_raise, qdotT_raise = self.create_transitional_modified(initial_player_pos, Tmove)
                    qT_lower, qdotT_lower = self.create_transitional_modified(final_player_pos, Tmove)
                    q6 = self.newton_raphson(final_player_pos)
                    
                    #self.seg_arr_msg.segments.append(create_seg(qT, v=qdotT, t=Tmove, gripper_val=GRIPPER_CLOSE_PURPLE))  # moving player position
                    self.seg_arr_msg.segments.append(create_seg(qT_raise, v=qdotT_raise, t=Tmove, gripper_val=GRIPPER_CLOSE_PURPLE))  # moving player position
                    self.seg_arr_msg.segments.append(create_seg(qT_lower, v=qdotT_lower, t=Tmove, gripper_val=GRIPPER_CLOSE_PURPLE))  # moving player position
                    self.seg_arr_msg.segments.append(create_seg(q6, t=Tmove, gripper_val=GRIPPER_CLOSE_PURPLE))  # placing player position

                    if self.down_snake == False and self.up_ladders == False:
                        self.seg_arr_msg.segments.append(create_seg(q6, t=Tmove))

                    if self.down_snake == True or self.up_ladders == True:

                        #qT2, qdotT2 = self.create_transitional(final_player_pos, snake_player_pos, Tmove)
                        qT2_raise, qdotT2_raise = self.create_transitional_modified(final_player_pos, Tmove)
                        qT2_lower, qdotT2_lower = self.create_transitional_modified(snake_player_pos, Tmove)
                        q7 = self.newton_raphson(snake_player_pos)
                        

                        #self.seg_arr_msg.segments.append(create_seg(qT2, v=qdotT2, t=Tmove, gripper_val=GRIPPER_CLOSE_PURPLE))  # moving player position
                        self.seg_arr_msg.segments.append(create_seg(qT2_raise, v=qdotT2_raise, t=Tmove, gripper_val=GRIPPER_CLOSE_PURPLE))  # moving player position
                        self.seg_arr_msg.segments.append(create_seg(qT2_lower, v=qdotT2_lower, t=Tmove, gripper_val=GRIPPER_CLOSE_PURPLE))  # moving player position
                        self.seg_arr_msg.segments.append(create_seg(q7, t=Tmove, gripper_val=GRIPPER_CLOSE_PURPLE))  # placing player position
                        self.seg_arr_msg.segments.append(create_seg(q7, t=Tmove))

                        self.down_snake = False
                        self.up_ladders = False

                self.seg_arr_msg.segments.append(create_seg(WAITING_POS, t=2*Tmove))
                self.seg_arr_msg.segments.append(create_seg(WAITING_POS, t=pi/4, gripper_val=GRIPPER_CLOSE_DICE))
                self.seg_arr_msg.segments.append(create_seg(WAITING_POS, t=pi/4))

                self.pub_segs.publish(self.seg_arr_msg)
                self.num_pub_player += 1
                self.check_board = False
                self.dice_hidden = False
                self.received_dice_roll = False
                self.recover_purple = 0

                self.seg_arr_msg.segments = []
            else:
                self.get_logger().debug('this sucks')    

    def recv_dice_box_array(self, msg):        
        if self.received_dice_roll == False and self.counter % 2 == 1 and self.num_pub_dice == 0:
            #self.get_logger().info('Dice Roll Hidden or Not: %s' % str(self.dice_hidden))
            if self.position != 1 and self.position != 100:
                if (abs(self.purple_disk_coords[0] - self.board_positions[self.position][0]) > 0.03\
                    or abs(self.purple_disk_coords[1] - self.board_positions[self.position][1]) > 0.03)\
                        and self.check_board == True and self.recover_purple == 0:
                    self.get_logger().debug('Purple player position: %s' % self.purple_disk_coords)
                    self.get_logger().debug('Expected position: %s' % [self.board_positions[self.position][0],
                                                                        self.board_positions[self.position][1], 0.012])
                    initial_player_pos = self.purple_disk_coords
                    final_player_pos = [self.board_positions[self.position][0], self.board_positions[self.position][1], 0.012]
                    Tmove = CYCLE / 2
                    qdisk = self.newton_raphson(initial_player_pos)
                    qdisk_raise = self.newton_raphson([self.purple_disk_coords[0], self.purple_disk_coords[1], self.purple_disk_coords[2] + 0.10])
                    qT_raise, qdotT_raise = self.create_transitional_modified(initial_player_pos, Tmove)
                    qT_lower, qdotT_lower = self.create_transitional_modified(final_player_pos, Tmove)
                    qfinal = self.newton_raphson(final_player_pos)
                    self.seg_arr_msg.segments.append(create_seg(qdisk_raise, t=2*Tmove))  # above player position
                    self.seg_arr_msg.segments.append(create_seg(qdisk, t=Tmove, gripper_val=GRIPPER_INTERMEDIATE))  # moving to player position
                    self.seg_arr_msg.segments.append(create_seg(qdisk, gripper_val=GRIPPER_CLOSE_PURPLE))  # gripping player position
                    self.seg_arr_msg.segments.append(create_seg(qT_raise, v=qdotT_raise, t=Tmove, gripper_val=GRIPPER_CLOSE_PURPLE))  # moving player position
                    self.seg_arr_msg.segments.append(create_seg(qT_lower, v=qdotT_lower, t=2*Tmove, gripper_val=GRIPPER_CLOSE_PURPLE))  # moving player position
                    self.seg_arr_msg.segments.append(create_seg(qfinal, t=Tmove, gripper_val=GRIPPER_CLOSE_PURPLE))  # placing player position
                    self.seg_arr_msg.segments.append(create_seg(qfinal, t=Tmove))
                    self.seg_arr_msg.segments.append(create_seg(WAITING_POS, t=2*Tmove))
                    self.seg_arr_msg.segments.append(create_seg(WAITING_POS, t=pi/4, gripper_val=GRIPPER_CLOSE_DICE))
                    self.seg_arr_msg.segments.append(create_seg(WAITING_POS, t=pi/4))
                    self.pub_segs.publish(self.seg_arr_msg)
                    self.seg_arr_msg.segments = []
                    self.num_pub_player += 1
                    self.recover_purple += 1
                    self.check_board = False
                    self.dice_hidden = False
                    self.counter = self.counter - 1
                    self.received_dice_roll = False
                else:
                    if (abs(self.curr_player_pos[0] - self.prev_player_pos[0]) > 0.01 or abs(self.curr_player_pos[1] - self.prev_player_pos[1]) > 0.01)\
                        and self.human_player_pos == 0:
                        human_player = 1
                        self.get_logger().debug('Human player position: %s' % human_player)
                        if (abs(self.curr_player_pos[0] - self.board_positions[human_player][0]) > 0.04 or\
                            abs(self.curr_player_pos[1] - self.board_positions[human_player][1]) > 0.04) and\
                                self.check_board == True:
                            self.get_logger().debug('CHEATING DETECTED!!!! RETURN PLAYER TO PROPER PLACE TO CONTINUE GAME')
                        elif self.check_board == True:
                            self.get_logger().debug('Rolling dice')
                            self.dice_face_msg.box = []
                            for box in msg.box:
                                self.dice_face_msg.box.append(box)
                            self.human_player_pos = human_player
                            Tmove = CYCLE / 2
                            dice_rest_pos = [self.dice_face_msg.box[0] + 0.02, self.dice_face_msg.box[1], 0.04] # self.dice_face_msg.box[0] + 0.025, self.dice_face_msg.box[1] - 0.01
                            #dice_rest_pos = [1.338, 0.301, 0.04]
                            lifted_dice_pos = [self.dice_face_msg.box[0] + 0.02, self.dice_face_msg.box[1], 0.11]
                            #lifted_dice_pos = [1.338, 0.301, 0.11]

                            q_dice_grip = self.newton_raphson(dice_rest_pos, J_dict_val='dice_bowl')
                            q_dice_drop = self.newton_raphson(lifted_dice_pos, J_dict_val='horizontal')
                            #q_dice_drop[3] = -np.pi/2
                            
                            self.seg_arr_msg.segments.append(create_seg(q_dice_grip, t=2*Tmove, gripper_val=GRIPPER_INTERMEDIATE))  # going to dice
                            self.seg_arr_msg.segments.append(create_seg(q_dice_grip, t=Tmove, gripper_val=GRIPPER_CLOSE_DICE))  # gripping the dice
                            self.seg_arr_msg.segments.append(create_seg(q_dice_drop, t=Tmove, gripper_val=GRIPPER_CLOSE_DICE))  # lifting dice up
                            self.seg_arr_msg.segments.append(create_seg(q_dice_drop, t=Tmove))  # dropping the dice
                            self.seg_arr_msg.segments.append(create_seg(WAITING_POS, t=2*Tmove))  # waiting 

                            self.pub_segs.publish(self.seg_arr_msg)
                            self.num_pub_dice += 1
                            self.check_board = False
                            self.seg_arr_msg.segments = []
                            self.dice_hidden = False
                            self.received_dice_roll = True
                    elif (abs(self.curr_player_pos[0] - self.prev_player_pos[0]) > 0.01 or abs(self.curr_player_pos[1] - self.prev_player_pos[1]) > 0.01)\
                        and self.human_player_pos != 0 and self.dice_hidden == True:
                        self.prev_human_player_pos = self.human_player_pos
                        human_player = self.prev_human_player_pos + self.dice_roll
                        if human_player in self.snakes:
                            human_player = self.snakes[human_player]
                        elif human_player in self.ladders:
                            human_player = self.ladders[human_player]
                        #self.get_logger().debug('Player Position: (%s, %s)' % (self.curr_player_pos[0], self.curr_player_pos[1]))
                        if human_player >= 100:
                            self.get_logger().debug('Human wins >:(')
                            human_player = 100
                            self.get_logger().debug('Human player position: %s' % human_player)
                            self.get_logger().debug('Human wins >:(')
                            self.human_player_pos = 0
                            self.counter = 0
                            self.check_board = True
                            self.reset = True
                            return
                        else:
                            self.get_logger().debug('Human player position: %s' % human_player)
                            if (abs(self.curr_player_pos[0] - self.board_positions[human_player][0]) > 0.04 or\
                                abs(self.curr_player_pos[1] - self.board_positions[human_player][1]) > 0.04) and\
                                    self.check_board == True:
                                self.get_logger().debug('CHEATING DETECTED!!!! RETURN PLAYER TO PROPER PLACE TO CONTINUE GAME')
                            elif self.check_board == True:
                                self.get_logger().debug('Rolling dice')
                                self.dice_face_msg.box = []
                                for box in msg.box:
                                    self.dice_face_msg.box.append(box)
                                self.human_player_pos = human_player
                                Tmove = CYCLE / 2
                                dice_rest_pos = [self.dice_face_msg.box[0] + 0.02, self.dice_face_msg.box[1], 0.04] # self.dice_face_msg.box[0] + 0.025, self.dice_face_msg.box[1] - 0.01
                                #dice_rest_pos = [1.338, 0.301, 0.04]
                                lifted_dice_pos = [self.dice_face_msg.box[0] + 0.02, self.dice_face_msg.box[1], 0.11]
                                #lifted_dice_pos = [1.338, 0.301, 0.11]

                                q_dice_grip = self.newton_raphson(dice_rest_pos, J_dict_val='dice_bowl')
                                q_dice_drop = self.newton_raphson(lifted_dice_pos, J_dict_val='horizontal')
                                #q_dice_drop[3] = -np.pi/2
                                
                                self.seg_arr_msg.segments.append(create_seg(q_dice_grip, t=2*Tmove, gripper_val=GRIPPER_INTERMEDIATE))  # going to dice
                                self.seg_arr_msg.segments.append(create_seg(q_dice_grip, t=Tmove, gripper_val=GRIPPER_CLOSE_DICE))  # gripping the dice
                                self.seg_arr_msg.segments.append(create_seg(q_dice_drop, t=Tmove, gripper_val=GRIPPER_CLOSE_DICE))  # lifting dice up
                                self.seg_arr_msg.segments.append(create_seg(q_dice_drop, t=Tmove))  # dropping the dice
                                self.seg_arr_msg.segments.append(create_seg(WAITING_POS, t=2*Tmove))  # waiting 

                                self.pub_segs.publish(self.seg_arr_msg)
                                self.num_pub_dice += 1
                                self.check_board = False
                                self.seg_arr_msg.segments = []
                                self.dice_hidden = False
                                self.received_dice_roll = True
                                self.recover_purple += 1
                    elif (abs(self.curr_player_pos[0] - self.prev_player_pos[0]) > 0.01 or abs(self.curr_player_pos[1] - self.prev_player_pos[1]) > 0.01)\
                        and self.human_player_pos !=0 and self.dice_hidden == False:
                        self.get_logger().debug('ROLL DICE TO CONTINUE GAME!!!')
                        self.recover_purple += 1
            elif self.position == 100:
                self.get_logger().debug('You LOSE :)!!')
                qdance = [1.13, 0.0, -pi/2, 0.0, 0.0]
                qT = [0.0, 0.0, -pi/2, 0.0, 0.0]
                qdance2 = [-1.2, 0.0, -pi/2, 0.0, 0.0]
                Tmove = CYCLE/2
                self.seg_arr_msg.segments.append(create_seg(qdance, t=Tmove))
                self.seg_arr_msg.segments.append(create_seg(qdance, t=Tmove/2, gripper_val=GRIPPER_CLOSE_DICE))
                self.seg_arr_msg.segments.append(create_seg(qdance2, t=Tmove))
                self.seg_arr_msg.segments.append(create_seg(qdance2, t=Tmove/2, gripper_val=GRIPPER_CLOSE_DICE))
                self.seg_arr_msg.segments.append(create_seg(qdance, t=Tmove))
                self.seg_arr_msg.segments.append(create_seg(qdance, t=Tmove/2, gripper_val=GRIPPER_CLOSE_DICE))
                self.seg_arr_msg.segments.append(create_seg(qdance2, t=Tmove))
                self.seg_arr_msg.segments.append(create_seg(qdance2, t=Tmove/2, gripper_val=GRIPPER_CLOSE_DICE))
                self.seg_arr_msg.segments.append(create_seg(WAITING_POS, t=Tmove/2))
                self.pub_segs.publish(self.seg_arr_msg)
                self.seg_arr_msg.segments = []
                self.num_pub_dice += 1
                self.received_dice_roll = True
                self.check_board = False
                self.dice_hidden = False
                self.recover_purple += 1
            else:
                if (abs(self.curr_player_pos[0] - self.prev_player_pos[0]) > 0.01 or abs(self.curr_player_pos[1] - self.prev_player_pos[1]) > 0.01)\
                        and self.human_player_pos == 0:
                        human_player = 1
                        self.get_logger().debug('Human player position: %s' % human_player)
                        if (abs(self.curr_player_pos[0] - self.board_positions[human_player][0]) > 0.04 or\
                            abs(self.curr_player_pos[1] - self.board_positions[human_player][1]) > 0.04) and\
                                self.check_board == True:
                            self.get_logger().debug('CHEATING DETECTED!!!! RETURN PLAYER TO PROPER PLACE TO CONTINUE GAME')
                        elif self.check_board == True:
                            self.get_logger().debug('Rolling dice')
                            self.dice_face_msg.box = []
                            for box in msg.box:
                                self.dice_face_msg.box.append(box)
                            self.human_player_pos = human_player
                            Tmove = CYCLE / 2
                            dice_rest_pos = [self.dice_face_msg.box[0] + 0.02, self.dice_face_msg.box[1], 0.04] # self.dice_face_msg.box[0] + 0.025, self.dice_face_msg.box[1] - 0.01
                            #dice_rest_pos = [1.338, 0.301, 0.04]
                            lifted_dice_pos = [self.dice_face_msg.box[0] + 0.02, self.dice_face_msg.box[1], 0.11]
                            #lifted_dice_pos = [1.338, 0.301, 0.11]

                            q_dice_grip = self.newton_raphson(dice_rest_pos, J_dict_val='dice_bowl')
                            q_dice_drop = self.newton_raphson(lifted_dice_pos, J_dict_val='horizontal')
                            #q_dice_drop[3] = -np.pi/2
                            
                            self.seg_arr_msg.segments.append(create_seg(q_dice_grip, t=2*Tmove, gripper_val=GRIPPER_INTERMEDIATE))  # going to dice
                            self.seg_arr_msg.segments.append(create_seg(q_dice_grip, t=Tmove, gripper_val=GRIPPER_CLOSE_DICE))  # gripping the dice
                            self.seg_arr_msg.segments.append(create_seg(q_dice_drop, t=Tmove, gripper_val=GRIPPER_CLOSE_DICE))  # lifting dice up
                            self.seg_arr_msg.segments.append(create_seg(q_dice_drop, t=Tmove))  # dropping the dice
                            self.seg_arr_msg.segments.append(create_seg(WAITING_POS, t=2*Tmove))  # waiting 

                            self.pub_segs.publish(self.seg_arr_msg)
                            self.num_pub_dice += 1
                            self.check_board = False
                            self.seg_arr_msg.segments = []
                            self.dice_hidden = False
                            self.received_dice_roll = True
                        # elif (abs(self.curr_player_pos[0] - self.prev_player_pos[0]) > 0.01 or abs(self.curr_player_pos[1] - self.prev_player_pos[1]) > 0.01)\
                        #     and self.human_player_pos != 0 and self.dice_hidden == True:
                        #     self.prev_human_player_pos = self.human_player_pos
                        #     human_player = self.prev_human_player_pos + self.dice_roll
                        #     if human_player in self.snakes:
                        #         human_player = self.snakes[human_player]
                        #     elif human_player in self.ladders:
                        #         human_player = self.ladders[human_player]
                        #     #self.get_logger().debug('Player Position: (%s, %s)' % (self.curr_player_pos[0], self.curr_player_pos[1]))
                        #     if human_player >= 100:
                        #         human_player = 100
                        #         self.get_logger().debug('Human player position: %s' % human_player)
                        #         self.get_logger().debug('Human wins >:(')
                        #         self.human_player_pos = 0
                        #         self.counter = 0
                        #         self.check_board = True
                        #         self.reset = True
                        #         return
                        #     else:
                        #         self.get_logger().debug('Human player position: %s' % human_player)
                        #         if (abs(self.curr_player_pos[0] - self.board_positions[human_player][0]) > 0.04 or\
                        #             abs(self.curr_player_pos[1] - self.board_positions[human_player][1]) > 0.04) and\
                        #                 self.check_board == True:
                        #             self.get_logger().debug('CHEATING DETECTED!!!! RETURN PLAYER TO PROPER PLACE TO CONTINUE GAME')
                        #         elif self.check_board == True:
                        #             self.get_logger().debug('Rolling dice')
                        #             self.dice_face_msg.box = []
                        #             for box in msg.box:
                        #                 self.dice_face_msg.box.append(box)
                        #             self.human_player_pos = human_player
                        #             Tmove = CYCLE / 2
                        #             dice_rest_pos = [self.dice_face_msg.box[0] + 0.02, self.dice_face_msg.box[1], 0.04] # self.dice_face_msg.box[0] + 0.025, self.dice_face_msg.box[1] - 0.01
                        #             #dice_rest_pos = [1.338, 0.301, 0.04]
                        #             lifted_dice_pos = [self.dice_face_msg.box[0] + 0.02, self.dice_face_msg.box[1], 0.11]
                        #             #lifted_dice_pos = [1.338, 0.301, 0.11]

                        #             q_dice_grip = self.newton_raphson(dice_rest_pos, J_dict_val='dice_bowl')
                        #             q_dice_drop = self.newton_raphson(lifted_dice_pos, J_dict_val='horizontal')
                        #             #q_dice_drop[3] = -np.pi/2
                                    
                        #             self.seg_arr_msg.segments.append(create_seg(q_dice_grip, t=2*Tmove, gripper_val=GRIPPER_INTERMEDIATE))  # going to dice
                        #             self.seg_arr_msg.segments.append(create_seg(q_dice_grip, t=Tmove, gripper_val=GRIPPER_CLOSE_DICE))  # gripping the dice
                        #             self.seg_arr_msg.segments.append(create_seg(q_dice_drop, t=Tmove, gripper_val=GRIPPER_CLOSE_DICE))  # lifting dice up
                        #             self.seg_arr_msg.segments.append(create_seg(q_dice_drop, t=Tmove))  # dropping the dice
                        #             self.seg_arr_msg.segments.append(create_seg(WAITING_POS, t=2*Tmove))  # waiting 

                        #             self.pub_segs.publish(self.seg_arr_msg)
                        #             self.num_pub_dice += 1
                        #             self.check_board = False
                        #             self.seg_arr_msg.segments = []
                        #             self.dice_hidden = False
                        #             self.received_dice_roll = True
                        # elif (abs(self.curr_player_pos[0] - self.prev_player_pos[0]) > 0.01 or abs(self.curr_player_pos[1] - self.prev_player_pos[1]) > 0.01)\
                        #     and self.human_player_pos !=0 and self.dice_hidden == False:
                        #     self.get_logger().debug('ROLL DICE TO CONTINUE GAME!!!')


    def recv_box_array(self, msg):
        if self.check_board == True:
            self.box_arr_msg.box = []
                
            for box in msg.box:
                self.box_arr_msg.box.append(box)
                
            w = 0.507 #.508
            h = 0.514
            # Calculate cell width & height (assuming a 10x10 board)
            cell_width = w / 10
            cell_height = h / 10    
            x_mid = self.box_arr_msg.box[0]
            y_mid = self.box_arr_msg.box[1]
            angle = self.box_arr_msg.box[2]

            #self.get_logger().info('Board Positions: %s, %s, %s' % (x_mid, y_mid, angle))
            
            self.board_positions = {}
            for row in range(5):
                for col in range(5):
                    x_pos_og = (x_mid - ((4 - col)*cell_width + cell_width/2))
                    y_pos_og = (y_mid - ((4 - row)*cell_height + cell_height/2)) 
                    x_pos = (x_pos_og - x_mid)*np.cos(np.radians(angle)) + (y_pos_og - y_mid)*np.sin(np.radians(angle)) + x_mid
                    y_pos = -(x_pos_og - x_mid)*np.sin(np.radians(angle)) + (y_pos_og - y_mid)*np.cos(np.radians(angle)) + y_mid
                    if row == 0:
                        cell_number = 91 + col
                    elif row == 1:
                        cell_number = 90 - col
                    elif row == 2:
                        cell_number = 71 + col
                    elif row == 3:
                        cell_number = 70 - col
                    elif row == 4:
                        cell_number = 51 + col
                    self.board_positions[cell_number] = (x_pos, y_pos)
            
            for row in range(5):
                for col in range(5, 10):
                    x_pos_og = x_mid + (col - 5) * cell_width + cell_width/2
                    y_pos_og = y_mid - (4 - row) * cell_height - cell_height/2
                    x_pos = (x_pos_og - x_mid)*np.cos(np.radians(angle)) + (y_pos_og - y_mid)*np.sin(np.radians(angle)) + x_mid
                    y_pos = -(x_pos_og - x_mid)*np.sin(np.radians(angle)) + (y_pos_og - y_mid)*np.cos(np.radians(angle)) + y_mid
                    if row == 0:
                        cell_number = 91 + col
                    elif row == 1:
                        cell_number = 90 - col
                    elif row == 2:
                        cell_number = 71 + col
                    elif row == 3:
                        cell_number = 70 - col
                    elif row == 4:
                        cell_number = 51 + col
                    self.board_positions[cell_number] = (x_pos, y_pos) #(x_pos + 0.022, y_pos - 0.02
            
            for row in range(5, 10):
                for col in range(5, 10):
                    x_pos_og = x_mid + (col - 5) * cell_width + cell_width/2
                    y_pos_og = y_mid + (row - 5) * cell_height + cell_height/2
                    x_pos = (x_pos_og - x_mid)*np.cos(np.radians(angle)) + (y_pos_og - y_mid)*np.sin(np.radians(angle)) + x_mid
                    y_pos = -(x_pos_og - x_mid)*np.sin(np.radians(angle)) + (y_pos_og - y_mid)*np.cos(np.radians(angle)) + y_mid
                    if row == 5:
                        cell_number = 50 - col 
                    elif row == 6:
                        cell_number = 31 + col
                    elif row == 7:
                        cell_number = 30 - col
                    elif row == 8:
                        cell_number = 11 + col
                    elif row == 9:
                        cell_number = 10 - col
                    self.board_positions[cell_number] = (x_pos, y_pos) #x_pos + 0.022, y_pos
            
            for row in range(5, 10):
                for col in range(5):
                    x_pos_og = x_mid - ((4 - col)*cell_width + cell_width/2)
                    y_pos_og = y_mid + ((row - 5) *cell_height + cell_height/2)
                    x_pos = (x_pos_og - x_mid)*np.cos(np.radians(angle)) + (y_pos_og - y_mid)*np.sin(np.radians(angle)) + x_mid
                    y_pos = -(x_pos_og - x_mid)*np.sin(np.radians(angle)) + (y_pos_og - y_mid)*np.cos(np.radians(angle)) + y_mid
                    if row == 5:
                        cell_number = 50 - col 
                    elif row == 6:
                        cell_number = 31 + col
                    elif row == 7:
                        cell_number = 30 - col
                    elif row == 8:
                        cell_number = 11 + col
                    elif row == 9:
                        cell_number = 10 - col

                    self.board_positions[cell_number] = (x_pos, y_pos)

            # self.get_logger().info('Board Positions: %s' % self.board_positions)

            # Define ladders manually (start → end)
            # self.ladders = {
            #     8: 27, 21: 41, 32: 51, 54: 66, 70: 89,
            #     77: 98
            # }
            self.ladders = {
                2: 19, 8: 27, 16: 37, 21: 41, 32: 51, 47: 68, 55: 66, 70: 89,
                86: 95, 78: 99
            }
            # Define snakes manually (start → end)
            # self.snakes = {
            #     15: 4, 29: 12, 46: 18, 68: 49, 79: 57,
            #     95: 74
            # }
            self.snakes = {
                15: 4, 29: 12, 46: 18, 33: 26, 59: 38, 69: 50, 79: 57,
                87: 66, 92: 71, 98: 77
            }
            
            if len(self.board_positions) != 0:
                self.obt_board_positions = True
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