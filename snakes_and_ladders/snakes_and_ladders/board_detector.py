import cv2
import numpy as np
import math

import rclpy
import cv_bridge

from rclpy.node         import Node
from sensor_msgs.msg    import Image

from project_msgs.msg import Object, ObjectArray



def board_detector(self, frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours (detect grid lines)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort and find the largest contour (assumed to be the board)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    board_contour = contours[0]  # Largest contour (the board)

    # Approximate board shape
    epsilon = 0.02 * cv2.arcLength(board_contour, True)
    approx = cv2.approxPolyDP(board_contour, epsilon, True)

    # Get bounding box of the board
    x, y, w, h = cv2.boundingRect(approx)

    return(x, y, w, h)
    
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
        
        # Assume the center of marker sheet is at the world origin.
        self.x0 = 0.664
        self.y0 = 0.455

        self.pubrgb = self.create_publisher(Image, name +'/image_raw', 3)
        
        self.get_logger().info("Name: %s" % name)

        self.bridge = cv_bridge.CvBridge()


        self.sub = self.create_subscription(
            Image, '/image_raw', self.process, 1)

    def shutdown(self):
        self.destroy_node()


        
    def process(self, msg):

        assert(msg.encoding == "rgb8")
        frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")

        # hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # binary = cv2.inRange(hsv, self.hsvlimits[:,0], self.hsvlimits[:,1])
        
        # (H, W, D) = frame.shape
        # uc = W//2
        # vc = H//2


        # iter = 2
        # binary = cv2.erode(binary, None, iterations=2*iter)
        # binary = cv2.dilate(binary, None, iterations=2*iter)
        # binary = cv2.erode(binary, None, iterations=2*iter)
       


        self.pubrgb.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))
        # self.pub_obj_array.publish(self.object_array)
        #self.pubbin.publish(self.bridge.cv2_to_imgmsg(binary))


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode('board_detector')
    rclpy.spin(node)
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()