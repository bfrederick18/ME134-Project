import cv2
import numpy as np
import math

import rclpy
import cv_bridge

from rclpy.node         import Node
from sensor_msgs.msg    import Image
from geometry_msgs.msg  import Point
from std_msgs.msg import Int16
from project_msgs.msg import Num, BoxArray
from snakes_and_ladders.constants import HSV_LIMITS_SIDECAM, MIN_DICE_FACE_AREA, MAX_DICE_FACE_AREA
        

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
        self.get_logger().info("Name: %s" % name)

        self.die_roll = 0
        self.counter = 0
        self.die_rolls = []
        
        self.dish_x = 0
        self.dish_y = 0

        self.M = None
        self.M2 = None
        self.dice_hidden = False
        self.initial_positions = {}
        self.dice_roll_rounded = 0
        self.dice_roll = Num()
        self.box_array = BoxArray()
        self.bridge = cv_bridge.CvBridge()
        self.dice_roll_hidden = Num()

        self.pubrgb = self.create_publisher(Image, name +'/image_raw', 3)
        self.pub_dice_face_gray = self.create_publisher(Image, name + '/dice_face_gray', 3)
        self.pub_dice_face_blurred = self.create_publisher(Image, name + '/dice_face_blurred', 3)
        self.pub_dice_face_thresh = self.create_publisher(Image, name + '/dice_face_thresh', 3)
        self.pub_dice_number_gray = self.create_publisher(Image, name + '/dice_number_gray', 3)
        self.pub_dice_number_roi = self.create_publisher(Image, name + '/dice_number_roi', 3)
        self.pub_dish_detector = self.create_publisher(Image, name + '/dish_detector', 3)
        self.pub_dish_hsv = self.create_publisher(Image, name +'/dish_hsv', 3)
        self.pub_dish_binary = self.create_publisher(Image, name +'/dish_binary', 3)
        self.dice_roll_pub = self.create_publisher(Num, name + '/num', 1)
        self.pub_box_array = self.create_publisher(BoxArray, name + '/box_array', 1)

        self.sub = self.create_subscription(
            Image, '/image_raw', self.process, 1)
        
        self.sub_dish_location = self.create_subscription(
            Point, '/board_detector/dish_location', self.recv_dish_location, 1)


    def shutdown(self):
        self.destroy_node()
    

    def recv_dish_location(self, msg):
        self.dish_x = msg.x + 0.065
        self.dish_y = msg.y


    def detect_dice_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 170, 250, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        self.pub_dice_face_gray.publish(self.bridge.cv2_to_imgmsg(gray))
        self.pub_dice_face_blurred.publish(self.bridge.cv2_to_imgmsg(blurred))
        self.pub_dice_face_thresh.publish(self.bridge.cv2_to_imgmsg(thresh))

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > MIN_DICE_FACE_AREA and area < MAX_DICE_FACE_AREA: 
                rotatedRectangle = cv2.minAreaRect(contour)
                ((x, y), (w, h), _) = rotatedRectangle
                box = np.int0(cv2.boxPoints(rotatedRectangle))
                cv2.drawContours(frame, [box], 0, (0, 0, 255), 1)
                return (x, y, w, h)


    def detect_dice_number(self, frame):
        dice_face = self.detect_dice_face(frame)
        if dice_face is None:
            self.get_logger().debug('Dice face not detected')
            return None
        
        x, y, w, h = dice_face
        w += 40
        h += 40
        w, h = max(1, int(w)), max(1, int(h))
        x1, y1 = max(0, int(x - w // 2)), max(0, int(y - h // 2))
        x2, y2 = min(frame.shape[1], int(x + w // 2)), min(frame.shape[0], int(y + h // 2))

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            self.get_logger().debug('ROI is empty')
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.0, 3, param1=25, param2=14, minRadius=4, maxRadius=7)  
        
        self.pub_dice_number_gray.publish(self.bridge.cv2_to_imgmsg(gray))

        if circles is None:
            self.get_logger().debug('Circles not detected')
            return None

        circles = np.round(circles[0, :]).astype('int')
        for (cx, cy, r) in circles:
            cv2.circle(roi, (cx, cy), r, (0, 255, 0), 2)
    
        self.pub_dice_number_roi.publish(self.bridge.cv2_to_imgmsg(roi, 'rgb8'))
        return len(circles)


    def detect_dice_dish(self, frame):
        dish_dectector_frame = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        binary = cv2.inRange(hsv, HSV_LIMITS_SIDECAM[:, 0], HSV_LIMITS_SIDECAM[:, 1])
        self.pub_dish_hsv.publish(self.bridge.cv2_to_imgmsg(hsv, 'rgb8'))

        iter = 2
        binary = cv2.erode(binary, None, iterations=iter)
        binary = cv2.dilate(binary, None, iterations=2*iter)
        binary = cv2.erode(binary, None, iterations=iter)

        self.pub_dish_binary.publish(self.bridge.cv2_to_imgmsg(binary))

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        dish_contour = contours[0]
        rotated_rectangle = cv2.minAreaRect(dish_contour)
        area = cv2.contourArea(dish_contour)

        if area >= 140000 and area <= 180000:
            box = np.int0(cv2.boxPoints(rotated_rectangle))
            cv2.drawContours(dish_dectector_frame, [box], -1, (0, 255, 0), 2)
            self.pub_dish_detector.publish(self.bridge.cv2_to_imgmsg(dish_dectector_frame, 'rgb8'))

            # Sort the box points to get a consistent order: top-left, top-right, bottom-right, bottom-left
            box = sorted(box, key=lambda x: (x[1], x[0]))

            def order_points(pts):
                pts = np.array(pts, dtype='float32')
                rect = np.zeros((4, 2), dtype='float32')
                s = pts.sum(axis=1)
                diff = np.diff(pts, axis=1)

                rect[0] = pts[np.argmin(s)]     # Top-left
                rect[2] = pts[np.argmax(s)]     # Bottom-right
                rect[1] = pts[np.argmin(diff)]  # Top-right
                rect[3] = pts[np.argmax(diff)]  # Bottom-left
                return rect
            
            ordered_box = order_points(box)
            
            act_dish_center_width = 0.124 / 2  # 0.13 shrunk bc sharpie
            act_dish_center_height = 0.12 / 2  # 0.13 shrunk bc sharpie

            dst_points = np.array([
                [self.dish_x - act_dish_center_width, self.dish_y - act_dish_center_height],  # Top-left
                [self.dish_x - act_dish_center_width, self.dish_y + act_dish_center_height],  # Top-right
                [self.dish_x + act_dish_center_width, self.dish_y + act_dish_center_height],  # Bottom-right
                [self.dish_x + act_dish_center_width, self.dish_y - act_dish_center_height]   # Bottom-left
            ], dtype=np.float32)

            self.M = cv2.getPerspectiveTransform(np.float32(ordered_box), dst_points) 
        

    def pixelToWorld(self, u, v, M):
        uvObj = np.float32([u, v])
        xyObj = cv2.perspectiveTransform(uvObj.reshape(1, 1, 2), M).reshape(2)    
        return xyObj

        
    def process(self, msg):
        #self.get_logger().info('this sucks')
        assert(msg.encoding == 'rgb8')
        frame = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        self.box_array.box = []
        # hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        # (H, W, D) = frame.shape
        # uc = W//2
        # vc = H//2
        # if True:
        #     # Draw the center lines.  Note the row is the first dimension.
        #     frame = cv2.line(frame, (uc,0), (uc,H-1), self.white, 1)
        #     frame = cv2.line(frame, (0,vc), (W-1,vc), self.white, 1)

        #     # Report the center HSV values.  Note the row comes first.
        #     self.get_logger().info(
        #         "HSV = (%3d, %3d, %3d)" % tuple(hsv[vc, uc]))

        dish = self.detect_dice_dish(frame)
        if dish is None:
            self.get_logger().info('Cant see dice dish')
        
        if type(self.M) is not np.ndarray:
            self.get_logger().debug('Calibration failed')
            return
        
        die_roll = self.detect_dice_number(frame)
        if die_roll is not None:
            self.die_rolls.append(die_roll)
            
            if len(self.die_rolls) > 50:
                self.die_rolls.pop(0)
            
            avg_reading = sum(self.die_rolls) / len(self.die_rolls)
            round_read = math.ceil(avg_reading)
            self.dice_roll_rounded = round_read

            self.dice_roll.num = self.dice_roll_rounded
            self.get_logger().info('Dice Reading: %s' % self.dice_roll_rounded)
            self.dice_roll_pub.publish(self.dice_roll)
        else:
            self.get_logger().info('NO DIE DETECTED!!!')
            self.die_rolls = []
            self.dice_hidden = True
            self.dice_roll_hidden.num = 0
            self.dice_roll_pub.publish(self.dice_roll_hidden)

        dice_frame = self.detect_dice_face(frame)

        if dice_frame is not None:
            x, y, _, _ = dice_frame
            dice_center_x, dice_center_y = self.pixelToWorld(int(x), int(y), self.M)
            if (dice_center_y <= 0.27 or dice_center_y >= 0.35) or (dice_center_x >= 1.39 or dice_center_x <= 1.31):
                self.box_array.box = [float(1.355), float(0.317)]
                self.pub_box_array.publish(self.box_array)
                self.get_logger().info('Dice Location: %s, %s' % (dice_center_x, dice_center_y))
            else:
                self.box_array.box = [float(dice_center_x), float(dice_center_y)]
                self.pub_box_array.publish(self.box_array)
                self.get_logger().info('Dice Location: %s, %s' % (dice_center_x, dice_center_y))
                
        else:
            self.get_logger().info('NO DIE DETECTED!!!')

        self.pubrgb.publish(self.bridge.cv2_to_imgmsg(frame, 'rgb8'))


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode('dice_detector')
    rclpy.spin(node)
    node.shutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()