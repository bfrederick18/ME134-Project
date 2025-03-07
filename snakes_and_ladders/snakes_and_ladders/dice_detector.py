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
        
        # Assume the center of marker sheet is at the world origin.
        self.x0 = 1.320 #1.338
        self.y0 = 0.326 #0.301
        
        self.x1 = 1.320  #dice bowl x position
        self.y1 = 0.326  #dice bowl y position

        self.dish_x = 0
        self.dish_y = 0

        self.M = None
        self.M2 = None
        self.initial_positions = {}
        self.dice_roll_rounded = 0
        self.dice_roll = Num()
        self.box_array = BoxArray()
        self.bridge = cv_bridge.CvBridge()

        self.pubrgb = self.create_publisher(Image, name +'/image_raw', 3)
        self.pub_dice_face_gray = self.create_publisher(Image, name + '/dice_face_gray', 3)
        self.pub_dice_face_blurred = self.create_publisher(Image, name + '/dice_face_blurred', 3)
        self.pub_dice_face_thresh = self.create_publisher(Image, name + '/dice_face_thresh', 3)
        self.pub_dice_number_gray = self.create_publisher(Image, name + '/dice_number_gray', 3)
        self.pub_dice_number_blurred = self.create_publisher(Image, name + '/dice_number_blurred', 3)
        self.pub_dice_number_roi = self.create_publisher(Image, name + '/dice_number_roi', 3)
        self.pub_dice_dish_gray = self.create_publisher(Image, name + '/dice_dish_gray', 3)
        self.pub_dice_dish_edge = self.create_publisher(Image, name + '/dice_dish_edge', 3)
        self.pub_roll = self.create_publisher(Int16, name + '/int16', 1)
        self.dice_roll_pub = self.create_publisher(Num, name + '/num', 1)
        self.pub_box_array = self.create_publisher(BoxArray, name + '/box_array', 1)

        self.sub = self.create_subscription(
            Image, '/image_raw', self.process, 1)
        
        self.sub_dish_location = self.create_subscription(
            Point, '/board_detector/dish_location', self.recv_dish_location, 1)


    def shutdown(self):
        self.destroy_node()
    

    def recv_dish_location(self, msg):
        self.dish_x = msg.x + 0.102
        self.dish_y = msg.y
        
        self.get_logger().info("Dish Location: %s, %s" % (self.dish_x, self.dish_y))


    def detect_dice_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # positive and odd kernel size

        # Threshold the image
        _, thresh = cv2.threshold(blurred, 170, 250, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        self.pub_dice_face_gray.publish(self.bridge.cv2_to_imgmsg(gray, "mono8"))
        self.pub_dice_face_blurred.publish(self.bridge.cv2_to_imgmsg(blurred, "mono8"))
        self.pub_dice_face_thresh.publish(self.bridge.cv2_to_imgmsg(thresh, "mono8"))

        # Filter contours and draw bounding box
        for contour in contours:
            area = cv2.contourArea(contour)
            #self.get_logger().info("Rotated Rectangle: %s" % area)
            if area > 500 and area < 2700:  # Adjust area threshold as needed
                rotatedRectangle = cv2.minAreaRect(contour)
                #self.get_logger().info("Rotated Rectangle: %s" % rotatedRectangle.__str__())
                ((x, y), (w, h), angle) = rotatedRectangle
                box = np.int0(cv2.boxPoints(rotatedRectangle))
                cv2.drawContours(frame, [box], 0, (0, 0, 255), 1)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                return (x, y, w, h)


    def detect_die_number(self, frame):
        # Detect the die face
        dice_face = self.detect_dice_face(frame)
        if dice_face is None:
            return None
        else:
            x, y, w, h = dice_face

            w += 20
            h += 20
            w, h = max(1, int(w)), max(1, int(h))

            # Compute ROI coordinates
            x1, y1 = max(0, int(x - w // 2)), max(0, int(y - h // 2))
            x2, y2 = min(frame.shape[1], int(x + w // 2)), min(frame.shape[0], int(y + h // 2))

            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return None

            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur
            # blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            # Apply Hough Circle Transform
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 0.5, 3,
                                    param1=30, param2=17, minRadius=3, maxRadius=10)
            
            self.pub_dice_number_gray.publish(self.bridge.cv2_to_imgmsg(gray))
            # self.pub_dice_number_blurred.publish(self.bridge.cv2_to_imgmsg(blurred, "mono8"))

            # Ensure at least some circles were found
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                # self.get_logger().info("Circle radiuses: %s" % circles)
                
                pip_centers = [(cx, cy) for (cx, cy, r) in circles]
                # Draw the circles
                for (cx, cy, r) in circles:
                    cv2.circle(roi, (cx, cy), r, (0, 255, 0), 2)
                

                if len(circles) == 5:
                    center_x = np.mean([p[0] for p in pip_centers])
                    center_y = np.mean([p[1] for p in pip_centers])
                    
                    distances = [((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5 for cx, cy in pip_centers]
                    
                    #If 4 points are at equal distance from the center, it's likely a cross
                    if np.std(distances) < 10:  # Adjust threshold as needed
                        return 5
                    else:
                        return 6

                                    
                self.pub_dice_number_roi.publish(self.bridge.cv2_to_imgmsg(roi, "rgb8"))
                return len(circles)
            else:
                return None


    def detect_dice_dish(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 20, 150)
        self.pub_dice_dish_edge.publish(self.bridge.cv2_to_imgmsg(edges, "mono8"))
        
        # Find contours and select the largest one (assumed to be the dish)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        if len(contours) == 0:
            return None  # No contours found
        
        dish_contour = contours[0]  # Largest contour (assumed to be the dish)

         # Get the minimum area rectangle
        rotated_rectangle = cv2.minAreaRect(dish_contour)
        ((um, vm), (wm, hm), angle) = rotated_rectangle
        
         # Get the 4 corner points of the rectangle
        box = np.int0(cv2.boxPoints(rotated_rectangle))  # Returns 4 points
        cv2.drawContours(gray, [box], -1, (0, 255, 0), 2)

        for contour in contours:
            a_rotated_rectangle = cv2.minAreaRect(contour)
            ((a_um, a_vm), (a_wm, a_hm), a_angle) = a_rotated_rectangle
            a_box = np.int0(cv2.boxPoints(a_rotated_rectangle))  # Returns 4 points
            # cv2.drawContours(gray, [a_box], -1, (0, 255, 0), 2)

        self.pub_dice_dish_gray.publish(self.bridge.cv2_to_imgmsg(gray, "mono8"))

         # Sort the box points to get a consistent order: top-left, top-right, bottom-right, bottom-left
        box = sorted(box, key=lambda x: (x[1], x[0]))  # Sort by y first, then x
        
        def order_points(pts):
            pts = np.array(pts, dtype="float32")
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)

            rect[0] = pts[np.argmin(s)]  # Top-left
            rect[2] = pts[np.argmax(s)]  # Bottom-right
            rect[1] = pts[np.argmin(diff)]  # Top-right
            rect[3] = pts[np.argmax(diff)]  # Bottom-left

            return rect

        ordered_box = order_points(box)
        
         #Actual W and H 
        dst_width = 0.13/2  
        dst_height = 0.13/2
        
        dst_points = np.array([
            [self.dish_x - dst_width, self.dish_y + dst_height],  # Top-left
            [self.dish_x + dst_width, self.dish_y + dst_height],  # Top-right
            [self.dish_x - dst_width, self.dish_y - dst_height],  # Bottom-right
            [self.dish_x + dst_width, self.dish_y - dst_height]  # Bottom-left
        ], dtype=np.float32)
        

        #Compute the perspective transformation matrix
        self.M = cv2.getPerspectiveTransform(np.float32(ordered_box), dst_points) 
        #self.get_logger().info('M: %s' % self.M)
    
    # def calibrate(self, image, x0, y0, annotateImage=True): 
    #     markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
    #         image, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    #     if annotateImage:
    #         cv2.aruco.drawDetectedMarkers(image, markerCorners, markerIds)

    #     if (markerIds is None or len(markerIds) != 4 or set(markerIds.flatten()) != set([1,2,3,4])):
    #         self.get_logger().debug('Not all markers detected')
    #         return None
        
    #     for i, marker_id in enumerate(markerIds.flatten()):
    #         center = np.mean(markerCorners[i], axis=1).flatten()
    #         self.initial_positions[marker_id] = center

    #     uvMarkers = np.zeros((4,2), dtype='float32')
    #     for i in range(4):
    #         uvMarkers[markerIds[i]-1,:] = np.mean(markerCorners[i], axis=1)

    #     DX = 0.105/2
    #     DY = 0.105/2
    #     xyMarkers = np.float32([
    #         [x0 - DX, y0 + DY],  # Top left
    #         [x0 + DX, y0 + DY],  # Top right
    #         [x0 - DX, y0 - DY],  # Bottom left
    #         [x0 + DX, y0 - DY]   # Bottom right
    #     ])       

    #     self.M = cv2.getPerspectiveTransform(uvMarkers, xyMarkers)
    #     self.get_logger().info('M: %s' % self.M)          

    def pixelToWorld(self, u, v, M):
        uvObj = np.float32([u, v])
        xyObj = cv2.perspectiveTransform(uvObj.reshape(1, 1, 2), M).reshape(2)    
        return xyObj

        
    def process(self, msg):
        assert(msg.encoding == "rgb8")
        frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        self.box_array.box = []

        #self.calibrate(frame, self.x0, self.y0, annotateImage=True)
        #self.detect_dice_dish(self, frame)
        M = self.detect_dice_dish(frame)
        
        if type(self.M) is not np.ndarray:
            self.get_logger().debug('Calibration failed')
            return
        
        die_roll = self.detect_die_number(frame)
        if die_roll is not None:
            self.die_rolls.append(die_roll)
            if len(self.die_rolls) == 40 and self.counter == 0:
                self.counter += 1
                avg_reading = average_list(self.die_rolls)
                round_read = math.ceil(avg_reading)
                self.dice_roll_rounded = round_read
                die_roll = self.detect_die_number(frame)
                self.die_rolls = []
            else:
                pass
        else:
            self.get_logger().info("NO DIE DETECTED!!!")
            self.die_rolls = []
            self.counter = 0
        
        dice_frame = self.detect_dice_face(frame)

        self.dice_roll.num = self.dice_roll_rounded
        self.get_logger().info("Dice Reading: %s" % self.dice_roll_rounded)
        self.dice_roll_pub.publish(self.dice_roll)

        if dice_frame is not None:
            x, y, w, h = dice_frame
            dice_center_x, dice_center_y = self.pixelToWorld(int(x), int(y), self.M)
            # dice_world = self.dish_x + dice_center_x, self.dish_y + dice_center_y
            self.box_array.box = [float(dice_center_x), float(dice_center_y)]
            self.pub_box_array.publish(self.box_array)
            self.get_logger().info("dice location: %s, %s" % (dice_center_x, dice_center_y))
        self.pubrgb.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))
        # self.pub_obj_array.publish(self.object_array)
        #self.pubbin.publish(self.bridge.cv2_to_imgmsg(binary))

def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode('dice_detector')
    rclpy.spin(node)
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()