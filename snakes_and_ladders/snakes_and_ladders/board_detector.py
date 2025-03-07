import cv2
import numpy as np
import math

import rclpy
import cv_bridge
import time

from rclpy.node         import Node
from sensor_msgs.msg    import Image
from geometry_msgs.msg  import Point

from project_msgs.msg import Object, ObjectArray, BoxArray

from snakes_and_ladders.constants import HSV_LIMITS_PURPLE, HSV_LIMITS_DISH, HSV_LIMITS_BLUE, LOGGER_LEVEL
 

def average_list(list):
    if not list:
        return 0
    return sum(list) / len(list)
            
            
class DetectorNode(Node):
    red    = (255,   0,   0)
    green  = (  0, 255,   0)
    blue   = (  0,   0, 255)
    yellow = (255, 255,   0)
    white  = (255, 255, 255)

    def __init__(self, name):
        super().__init__(name)
        self.get_logger().set_level(LOGGER_LEVEL)
        self.get_logger().info('Name: %s' % name)
        
        # Assume the center of marker sheet is at the world origin.
        self.x0 = 0.6385  # self.x0 = 0.5805, 0.6285  # + an additional 4.8cm in x
        self.y0 = 0.3755   # self.y0 = 0.3320, 0.379  # and an additional 4.7cm in y

        self.x1 = 1.320  #dice bowl x position
        self.y1 = 0.326  #dice bowl y position
        
        self.initial_positions = {}
        self.initial_dice_marker_positions = {}
        self.M = None
        self.M2 = None
        self.object_array = ObjectArray()
        self.box_array = BoxArray()
        self.bridge = cv_bridge.CvBridge()
        
        self.pub_rgb = self.create_publisher(Image, name +'/image_raw', 3)
        self.pub_binary = self.create_publisher(Image, name +'/binary', 3)
        self.pub_board = self.create_publisher(Image, name +'/board', 3)
        self.pub_dish_hsv = self.create_publisher(Image, name +'/dish_hsv', 3)
        self.pub_dish_binary = self.create_publisher(Image, name +'/dish_binary', 3)
        self.pub_dish_detector = self.create_publisher(Image, name +'/dish_detector', 3)
        self.pub_obj_array = self.create_publisher(ObjectArray, name + '/object_array', 1)
        self.pub_box_array = self.create_publisher(BoxArray, name + '/box_array', 1)

        self.pub_dish_location = self.create_publisher(Point, name + '/dish_location', 1)

        self.sub = self.create_subscription(
            Image, '/image_raw', self.process, 1)
        
        self.counter = 0
        
        self.get_logger().info('Board detector running...')


    def shutdown(self):
        self.destroy_node()


    def board_detector(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        board_contour = contours[0]  # Largest contour (the board)

        rotated_rectangle = cv2.minAreaRect(board_contour)
        ((um, vm), (wm,  hm), angle) = cv2.minAreaRect(board_contour)
        
        box = np.int0(cv2.boxPoints(rotated_rectangle))
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

        # self.pub_rgb.publish(self.bridge.cv2_to_imgmsg(frame, 'rgb8'))
        self.pub_board.publish(self.bridge.cv2_to_imgmsg(edges))
                
        #to deal with CW verus CCW Rotation
        if angle < 45.0 and angle >= 0.0:
            angle = angle
        elif angle <= 90.0 and angle >= 45.0:
            angle = angle - 90
        
        self.get_logger().info('Angle: %s' % angle)
            
        return(um, vm, wm, hm, angle)   
    

    def dish_detector(self, frame):
        dish_dectector_frame = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        binary = cv2.inRange(hsv, HSV_LIMITS_DISH[:, 0], HSV_LIMITS_DISH[:, 1])
        self.pub_dish_hsv.publish(self.bridge.cv2_to_imgmsg(hsv, 'rgb8'))
        # self.pub_dish_binary.publish(self.bridge.cv2_to_imgmsg(binary))

        iter = 2
        binary = cv2.erode(binary, None, iterations=iter)
        binary = cv2.dilate(binary, None, iterations=2*iter)
        binary = cv2.erode(binary, None, iterations=iter)

        self.pub_dish_binary.publish(self.bridge.cv2_to_imgmsg(binary))

        (contours, hierarchy) = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(dish_dectector_frame, contours, -1, self.blue, 1)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(dish_dectector_frame, (x, y), (x + w, y + h), self.red, 2)
            self.pub_dish_detector.publish(self.bridge.cv2_to_imgmsg(dish_dectector_frame, 'rgb8'))
            return x, y, w, h

    
    def calibrate(self, image, x0, y0, annotateImage=True): 
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
            image, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        if annotateImage:
            cv2.aruco.drawDetectedMarkers(image, markerCorners, markerIds)

        VALID_IDS = {1, 2, 3, 4}
        if markerIds is not None:
            filtered_corners = []
            filtered_ids = []

            for i in range(len(markerIds)):
                marker_id = markerIds[i][0]
                if marker_id in VALID_IDS:  # Keep only selected IDs
                    filtered_corners.append(markerCorners[i])
                    filtered_ids.append([marker_id])

            # Convert filtered lists back to NumPy arrays
            if filtered_ids:
                filtered_ids = np.array(filtered_ids)
                markerIds = filtered_ids
                markerCorners = filtered_corners
            else:
                filtered_ids = None
 
        if (markerIds is None or len(markerIds) != 4 or set(markerIds.flatten()) != set([1,2,3,4])):
            self.get_logger().debug('Markers Detected: %s' % set(markerIds.flatten()))
            return None
        
        for i, marker_id in enumerate(markerIds.flatten()):
            center = np.mean(markerCorners[i], axis=1).flatten()
            self.initial_positions[marker_id] = center
        
        uvMarkers = np.zeros((4,2), dtype='float32')
        for i in range(4):
            uvMarkers[markerIds[i]-1,:] = np.mean(markerCorners[i], axis=1)

        DX = 1.175/2
        DY = 0.653/2
        xyMarkers = np.float32([
            [x0 - DX, y0 + DY],  # Top left
            [x0 + DX, y0 + DY],  # Top right
            [x0 - DX, y0 - DY],  # Bottom left
            [x0 + DX, y0 - DY]   # Bottom right
        ])       

        self.M = cv2.getPerspectiveTransform(uvMarkers, xyMarkers)

    def pixelToWorld(self, u, v, M):
        uvObj = np.float32([u, v])
        xyObj = cv2.perspectiveTransform(uvObj.reshape(1, 1, 2), M).reshape(2)
        return xyObj
    

    def player_detector(self, frame, hsv_limits, object_type):
        self.get_logger().info('Detecting %s' % object_type)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        binary = cv2.inRange(hsv, hsv_limits[:, 0], hsv_limits[:, 1])

        iter = 2
        binary = cv2.erode(binary, None, iterations=iter)
        binary = cv2.dilate(binary, None, iterations=2*iter)
        binary = cv2.erode(binary, None, iterations=iter)

        (contours, hierarchy) = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(frame, contours, -1, self.blue, 1)

        if len(contours) > 0:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            ((u, v), radius) = cv2.minEnclosingCircle(contours[0])
            u = int(u)
            v = int(v)
            radius = int(radius)

            try:
                ellipse = cv2.fitEllipse(contours[0])
                ((ue, ve), (we, he), angle) = ellipse
            except Exception as e:
                ellipse = None

            if ellipse is not None:
                cv2.ellipse(frame, ellipse, self.green, 2)
                obj_player_world = self.pixelToWorld(int(ue), int(ve), self.M)
                if obj_player_world is not None:
                    obj_player_world_x, obj_player_world_y = obj_player_world
                    obj_player = Object()
                    obj_player.type = object_type
                    obj_player.x = float(obj_player_world_x)
                    obj_player.y = float(obj_player_world_y)
                    obj_player.z = 0.0
                    obj_player.theta = 0.0
                    self.object_array.objects.append(obj_player)

        if object_type == Object.PURPLE_DISK:
            self.pub_binary.publish(self.bridge.cv2_to_imgmsg(binary))


    def process(self, msg):
        self.object_array.objects = []
        self.box_array.box = []

        assert(msg.encoding == 'rgb8')
        frame = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

        dish_x, dish_y, dish_w, dish_h = self.dish_detector(frame)

        old_M = self.M
        self.calibrate(frame, self.x0, self.y0, annotateImage=True)

        if type(self.M) is not np.ndarray:
            self.get_logger().debug('Calibration failed')
            self.pub_rgb.publish(self.bridge.cv2_to_imgmsg(frame, 'rgb8'))
            return
        elif old_M is not None and not np.allclose(self.M, old_M):
            self.get_logger().info('Calibration updated')
            pass

        self.player_detector(frame, HSV_LIMITS_PURPLE, Object.PURPLE_DISK)
        self.player_detector(frame, HSV_LIMITS_BLUE, Object.BLUE_DISK)

        [um, vm, wm, hm, angle] = self.board_detector(frame)
        board_center_x, board_center_y = self.pixelToWorld(int(um), int(vm), self.M)

        self.box_array.box = [float(board_center_x), float(board_center_y), float(angle)]

        dish_right_edge_center = self.pixelToWorld(dish_x + dish_w, dish_y + dish_h/2, self.M)
        self.get_logger().info('Dish Right Edge Center: %s' % str(dish_right_edge_center))
        dish_point = Point()
        dish_point.x = (float)(dish_right_edge_center[0])
        dish_point.y = (float)(dish_right_edge_center[1])
        dish_point.z = 0.0
        self.pub_dish_location.publish(dish_point)
            
        self.pub_rgb.publish(self.bridge.cv2_to_imgmsg(frame, 'rgb8'))
        self.pub_obj_array.publish(self.object_array)
        self.pub_box_array.publish(self.box_array)


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode('board_detector')
    rclpy.spin(node)
    node.shutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()