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

    # Get bounding box of the board
    rotated_rectangle = cv2.minAreaRect(board_contour)
    ((um, vm), (wm,  hm), angle) = cv2.minAreaRect(board_contour)
    box = np.int0(cv2.boxPoints(rotated_rectangle))
    cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

    return(um, vm, wm, hm)
    
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

        self.hsvlimits = np.array([[100, 130], [100, 130], [70, 110]])
        
        # Assume the center of marker sheet is at the world origin.
        self.x0 = 0.664
        self.y0 = 0.455

        self.pubrgb = self.create_publisher(Image, name +'/image_raw', 3)

        self.pub_obj_array = self.create_publisher(ObjectArray, name + '/object_array', 1)
        self.object_array = ObjectArray()
        
        self.get_logger().info("Name: %s" % name)

        self.bridge = cv_bridge.CvBridge()


        self.sub = self.create_subscription(
            Image, '/image_raw', self.process, 1)

    def shutdown(self):
        self.destroy_node()


        
    def process(self, msg):

        assert(msg.encoding == "rgb8")
        frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")

        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        binary = cv2.inRange(hsv, self.hsvlimits[:,0], self.hsvlimits[:,1])

        (H, W, D) = frame.shape
        uc = W//2
        vc = H//2

        # # Help to determine the HSV range...
        # if True:
        #     # Draw the center lines.  Note the row is the first dimension.
        #     frame = cv2.line(frame, (uc,0), (uc,H-1), self.white, 1)
        #     frame = cv2.line(frame, (0,vc), (W-1,vc), self.white, 1)

        #     # Report the center HSV values.  Note the row comes first.
        #     self.get_logger().info(
        #         "HSV = (%3d, %3d, %3d)" % tuple(hsv[vc, uc]))

        iter = 2
        binary = cv2.erode(binary, None, iterations=iter)
        binary = cv2.dilate(binary, None, iterations=2*iter)
        binary = cv2.erode(binary, None, iterations=iter)

        (contours, hierarchy) = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(frame, contours, -1, self.blue, 1)

        if len(contours) > 0:
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                ((ur, vr), radius) = cv2.minEnclosingCircle(contour)
                ur     = int(ur)
                vr     = int(vr)
                radius = int(radius)

                try:
                    ellipse = cv2.fitEllipse(contour)
                    ((ue, ve), (we, he), angle) = ellipse
                except Exception as e:
                    self.get_logger().info("Exception: %s" % str(e))
                    ellipse = None

                if ellipse is not None:
                    cv2.ellipse(frame, ellipse, self.green, 1)
                    #cv2.circle(frame, (int(ue), int(ve)), 5, self.red, -1)

        board_detector(self, frame)

        self.pubrgb.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))
        self.pub_obj_array.publish(self.object_array)
        #self.pubbin.publish(self.bridge.cv2_to_imgmsg(binary))


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode('board_detector')
    rclpy.spin(node)
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()