import cv2
import numpy as np
import math

import rclpy
import cv_bridge

from rclpy.node         import Node
from sensor_msgs.msg    import Image
from std_msgs.msg import Int16


def detect_die_number(self, frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur and threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 0.5, 3,
                            param1=30, param2=17, minRadius=1, maxRadius=10) #(50, 20)

    # Ensure at least some circles were found
    if circles is not None:
        # Convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # Draw the circles
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
        
        return(len(circles))
    else:
        return None


def detect_dice_face(self, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image
    _, thresh = cv2.threshold(blurred, 170, 250, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filter contours and draw bounding box
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Adjust area threshold as needed
            rotatedRectangle = cv2.minAreaRect(contour)
            ((x, y), (w, h), angle) = cv2.minAreaRect(contour)
            box = np.int0(cv2.boxPoints(rotatedRectangle))
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    

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

        #self.hsvlimits = np.array([[10, 40], [60, 220], [125, 255]])

        self.die_roll = 0
        self.counter = 0
        self.die_rolls = []
        
        # Assume the center of marker sheet is at the world origin.
        self.x0 = 1.320
        self.y0 = 0.285

        self.pubrgb = self.create_publisher(Image, name +'/image_raw', 3)
        
        self.get_logger().info("Name: %s" % name)

        self.bridge = cv_bridge.CvBridge()

        self.pub_roll = self.create_publisher(Int16, name + '/int16', 1)

        self.sub = self.create_subscription(
            Image, '/image_raw', self.process, 1)


    def shutdown(self):
        self.destroy_node()

        
    def process(self, msg):
        assert(msg.encoding == "rgb8")
        frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        
        die_roll = detect_die_number(self,frame)
        if die_roll is not None:
            self.die_rolls.append(die_roll)
            if len(self.die_rolls) == 50 and self.counter == 0:
                self.counter += 1
                avg_reading = average_list(self.die_rolls)
                round_read = math.ceil(avg_reading)
                self.get_logger().info("Dice Reading: %s" % round_read)
                die_roll = detect_die_number(self,frame)
                self.die_rolls = []
                #self.pub_roll.publish(die_roll)
            else:
                pass
        else:
            self.get_logger().info("NO DIE DETECTED!!!")
            self.die_rolls = []
            self.counter = 0

        detect_dice_face(self, frame)

        self.pubrgb.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))
        # self.pub_obj_array.publish(self.object_array)
        #self.pubbin.publish(self.bridge.cv2_to_imgmsg(binary))
        
    def calibrate(self, image, x0, y0, annotateImage=True): 
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
            image, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        if annotateImage:
            cv2.aruco.drawDetectedMarkers(image, markerCorners, markerIds)

        if (markerIds is None or len(markerIds) != 4 or set(markerIds.flatten()) != set([1,2,3,4])):
            self.get_logger().debug('Not all markers detected')
            return None
        
        for i, marker_id in enumerate(markerIds.flatten()):
            center = np.mean(markerCorners[i], axis=1).flatten()
            self.initial_positions[marker_id] = center

        uvMarkers = np.zeros((4,2), dtype='float32')
        for i in range(4):
            uvMarkers[markerIds[i]-1,:] = np.mean(markerCorners[i], axis=1)

        DX = 0.112/2
        DY = 0.112/2
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

def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode('dice_detector')
    rclpy.spin(node)
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()