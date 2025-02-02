#!/usr/bin/env python3
#
#   balldetector.py
#
#   Detect the tennis balls with OpenCV.
#
#   Node:           /balldetector
#   Subscribers:    /usb_cam/image_raw          Source image
#   Publishers:     /balldetector/binary        Intermediate binary image
#                   /balldetector/image_raw     Debug (marked up) image
#
import cv2
import numpy as np

import rclpy
import cv_bridge

from rclpy.node         import Node
from sensor_msgs.msg    import Image

from project_msgs.msg import Object, ObjectArray


TAP_FACTOR = 4.5


class DetectorNode(Node):
    red    = (255,   0,   0)
    green  = (  0, 255,   0)
    blue   = (  0,   0, 255)
    yellow = (255, 255,   0)
    white  = (255, 255, 255)

    def __init__(self, name):
        super().__init__(name)

        self.hsvlimits = np.array([[10, 40], [60, 220], [125, 255]])

        self.pubrgb = self.create_publisher(Image, name +'/image_raw', 3)
        self.pubbin = self.create_publisher(Image, name +'/binary',    3)

        self.pub_obj_array = self.create_publisher(ObjectArray, name + '/object_array', 1)
        
        self.get_logger().info("Name: %s" % name)

        self.bridge = cv_bridge.CvBridge()

        self.object_array = ObjectArray()
        self.object_array.objects = []

        self.sub = self.create_subscription(
            Image, '/image_raw', self.process, 1)

        # Report.
        self.get_logger().info("Ball detector running...")

    def shutdown(self):
        self.destroy_node()


    # Process the image (detect the ball).
    def process(self, msg):
        self.object_array.objects = []

        assert(msg.encoding == "rgb8")
        frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")

        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        (H, W, D) = frame.shape
        uc = W//2
        vc = H//2

        # Help to determine the HSV range...
        if False:
            # Draw the center lines.  Note the row is the first dimension.
            # frame = cv2.line(frame, (uc,0), (uc,H-1), self.white, 1)
            frame = cv2.line(frame, (30,0), (30,H-1), self.white, 1)

            # frame = cv2.line(frame, (0,vc), (W-1,vc), self.white, 1)
            frame = cv2.line(frame, (0,vc+60), (W-1,vc+60), self.white, 1)

            # Report the center HSV values.  Note the row comes first.
            # self.get_logger().info("HSV = (%3d, %3d, %3d)" % tuple(hsv[vc+60, 30]))

        binary = cv2.inRange(hsv, self.hsvlimits[:,0], self.hsvlimits[:,1])

        iter = 2
        binary = cv2.erode( binary, None, iterations=iter)
        binary = cv2.dilate(binary, None, iterations=2*iter)
        binary = cv2.erode( binary, None, iterations=iter)

        (contours, hierarchy) = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(frame, contours, -1, self.blue, 2)

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
                    if he > we * 2:
                        rotatedRect = cv2.minAreaRect(contour)
                        ((um, vm), (wm, hm), angle) = cv2.minAreaRect(contour)

                        if wm < hm:
                            angle += 90
                            (wm, hm) = (hm, wm)

                        box = np.int0(cv2.boxPoints(rotatedRect))
                        cv2.drawContours(frame, [box], 0, self.green, 2)

                        cv2.line(frame, 
                                (int(um - TAP_FACTOR * (hm / 2) * np.sin(np.radians(angle))), int(vm + TAP_FACTOR * (hm / 2) * np.cos(np.radians(angle)))), 
                                (int(um + TAP_FACTOR * (hm / 2) * np.sin(np.radians(angle))), int(vm - TAP_FACTOR * (hm / 2) * np.cos(np.radians(angle)))), 
                                self.yellow, 2)
                        cv2.circle(frame, (int(um), int(vm)), 5, self.red,    -1)

                        obj_rect = Object()
                        obj_rect.type = Object.STRIP
                        obj_rect.x = float(um)
                        obj_rect.y = float(vm)
                        obj_rect.z = 0.0
                        obj_rect.theta = angle

                        self.object_array.objects.append(obj_rect)

                        self.get_logger().info("angle: %s" % str(angle))

                    else:
                        cv2.ellipse(frame, ellipse, self.green, 2)
                        cv2.circle(frame, (int(ue), int(ve)), 5, self.red,    -1)

                        obj_disk = Object()
                        obj_disk.type = Object.DISK
                        obj_disk.x = ue
                        obj_disk.y = ve
                        obj_disk.z = 0.0
                        obj_disk.theta = 0.0

                        self.object_array.objects.append(obj_disk)

        self.pubrgb.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))
        self.pub_obj_array.publish(self.object_array)
        self.pubbin.publish(self.bridge.cv2_to_imgmsg(binary))


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode('detector')
    rclpy.spin(node)
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()