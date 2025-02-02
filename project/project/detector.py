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

#from geometry_msgs.msg  import Point, Pose
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

        # self.hsvlimits = np.array([[20, 30], [90, 170], [60, 255]])
        # self.hsvlimits = np.array([[0, 30], [160, 250], [20, 255]])  
        # self.hsvlimits = np.array([[10, 40], [60, 200], [20, 255]])
        self.hsvlimits = np.array([[10, 40], [60, 220], [125, 255]])

        self.pubrgb = self.create_publisher(Image, name+'/image_raw', 3)
        self.pubbin = self.create_publisher(Image, name+'/binary',    3)

        self.pub_obj_array = self.create_publisher(ObjectArray, name + '/object_array', 1)
        
        # self.pub_point_start = self.create_publisher(Point, name + '/strip_pose_start', 1)
        # self.pub_point_end = self.create_publisher(Point, name + '/strip_pose_end', 1)
        # self.pub_point_center = self.create_publisher(Point, name + '/strip_pose_center', 1)

        self.get_logger().info("Name: %s" % name)

        self.bridge = cv_bridge.CvBridge()

        self.object_list = ObjectArray()

        # Finally, subscribe to the incoming image topic.  Using a
        # queue size of one means only the most recent message is
        # stored for the next subscriber callback.
        self.sub = self.create_subscription(
            Image, '/image_raw', self.process, 1)

        # Report.
        self.get_logger().info("Ball detector running...")

    # Shutdown
    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.destroy_node()


    # Process the image (detect the ball).
    def process(self, msg):
        # Confirm the encoding and report.
        assert(msg.encoding == "rgb8")
        # self.get_logger().info(
        #     "Image %dx%d, bytes/pixel %d, encoding %s" %
        #     (msg.width, msg.height, msg.step/msg.width, msg.encoding))

        # Convert into OpenCV image, using RGB 8-bit (pass-through).
        frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Cheat: swap red/blue

        # Grab the image shape, determine the center pixel.
        (H, W, D) = frame.shape
        uc = W//2
        vc = H//2

        # Help to determine the HSV range...
        if True:
            # Draw the center lines.  Note the row is the first dimension.
            # frame = cv2.line(frame, (uc,0), (uc,H-1), self.white, 1)
            frame = cv2.line(frame, (30,0), (30,H-1), self.white, 1)

            # frame = cv2.line(frame, (0,vc), (W-1,vc), self.white, 1)
            frame = cv2.line(frame, (0,vc+60), (W-1,vc+60), self.white, 1)

            # Report the center HSV values.  Note the row comes first.
            # self.get_logger().info("HSV = (%3d, %3d, %3d)" % tuple(hsv[vc+60, 30]))

        
        # Threshold in Hmin/max, Smin/max, Vmin/max
        binary = cv2.inRange(hsv, self.hsvlimits[:,0], self.hsvlimits[:,1])

        # Erode and Dilate. Definitely adjust the iterations!
        iter = 2
        binary = cv2.erode( binary, None, iterations=iter)
        binary = cv2.dilate(binary, None, iterations=2*iter)
        binary = cv2.erode( binary, None, iterations=iter)


        # Find contours in the mask and initialize the current
        # (x, y) center of the ball
        (contours, hierarchy) = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw all contours on the original image for debugging.
        cv2.drawContours(frame, contours, -1, self.blue, 2)

        # Only proceed if at least one contour was found.  You may
        # also want to loop over the contours...
        if len(contours) > 0:
            # Pick the largest contour.
            # contour = max(contours, key=cv2.contourArea)
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            # Find the enclosing circle (convert to pixel values)
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

                # self.get_logger().info("Ellipse: %s" % str(ellipse))
                if ellipse is not None:
                    if he > we * 2:
                        rotatedRect = cv2.minAreaRect(contour)
                        ((um, vm), (wm, hm), angle) = cv2.minAreaRect(contour)

                        if wm < hm:
                            angle += 90
                            (wm, hm) = (hm, wm)

                        box = np.int0(cv2.boxPoints(rotatedRect))
                        cv2.drawContours(frame, [box], 0, self.green, 2)

                        #cv2.line(frame, 
                                #(int(um - (wm / 2) * np.cos(np.radians(angle))), int(vm - (wm / 2) * np.sin(np.radians(angle)))), 
                                #(int(um + (wm / 2) * np.cos(np.radians(angle))), int(vm + (wm / 2) * np.sin(np.radians(angle)))), 
                                #self.yellow, 2)
                        
                        cv2.line(frame, 
                                (int(um - TAP_FACTOR * (hm / 2) * np.sin(np.radians(angle))), int(vm + TAP_FACTOR * (hm / 2) * np.cos(np.radians(angle)))), 
                                (int(um + TAP_FACTOR * (hm / 2) * np.sin(np.radians(angle))), int(vm - TAP_FACTOR * (hm / 2) * np.cos(np.radians(angle)))), 
                                self.yellow, 2)
                        cv2.circle(frame, (int(um), int(vm)), 5, self.red,    -1)

                        rect_start = [int(um - TAP_FACTOR * (hm / 2) * np.sin(np.radians(angle))), int(vm + TAP_FACTOR * (hm / 2) * np.cos(np.radians(angle)))]
                        rect_end = [int(um + TAP_FACTOR * (hm / 2) * np.sin(np.radians(angle))), int(vm - TAP_FACTOR * (hm / 2) * np.cos(np.radians(angle)))]
                        
                        
                        # rect_point_start_msg = Point()
                        # rect_point_start_msg.x = float(rect_start[0])
                        # rect_point_start_msg.y = float(rect_start[1])
                        # rect_point_start_msg.z = 0.0
                        # self.pub_point_start.publish(rect_point_start_msg)
                        
                        # rect_point_end_msg = Point()
                        # rect_point_end_msg.x = float(rect_end[0])
                        # rect_point_end_msg.y = float(rect_end[1])
                        # rect_point_end_msg.z = 0.0
                        # self.pub_point_end.publish(rect_point_end_msg)

                        # rect_point_center_msg = Point()
                        # rect_point_center_msg.x = float(um)
                        # rect_point_center_msg.y = float(vm)
                        # rect_point_center_msg.z = 0.0
                        # self.pub_point_center.publish(rect_point_center_msg)
                        
                        # self.get_logger().info("Rect: %s" % str(rect_point_msg))

                    else:
                        cv2.ellipse(frame, ellipse, self.green, 2)
                        cv2.circle(frame, (int(ue), int(ve)), 5, self.red,    -1)

                        # disc_point_msg = Point()
                        # disc_point_msg.x = ue
                        # disc_point_msg.y = ve
                        # disc_point_msg.z = 0.0

                        obj_disk = Object()
                        obj_disk.type = Object.DISK
                        obj_disk.pose.position.x = ue
                        obj_disk.pose.position.y = ve
                        obj_disk.pose.position.z = 0.0
                        obj_disk.pose.quaternion.x = 0.0
                        obj_disk.pose.quaternion.y = 0.0
                        obj_disk.pose.quaternion.z = 0.0
                        obj_disk.pose.quaternion.w = 0.0

                        self.object_list.append(obj_disk)


                        #self.pub_point.publish(disc_point_msg)
                        self.get_logger().info("Disc: %s" % str(obj_disk))

                    # Draw the circle (yellow) and centroid (red) on the
                    # original image.
                    #cv2.circle(frame, (ur, vr), int(radius), self.yellow,  2)
                    # cv2.circle(frame, (int(ue), int(ve)), 5, self.red,    -1)

            # Report.
            # self.get_logger().info(
            #     "Found Ball enclosed by radius %d about (%d,%d)" %
            #     (radius, ur, vr))

        # Convert the frame back into a ROS image and republish.
        self.pubrgb.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))

        self.pub_point.publish(self.object_list)

        # Also publish the binary (black/white) image.
        self.pubbin.publish(self.bridge.cv2_to_imgmsg(binary))


def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the detector node.
    node = DetectorNode('detector')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()