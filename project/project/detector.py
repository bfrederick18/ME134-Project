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

        # Assume the center of marker sheet is at the world origin.
        self.x0 = 0.664
        self.y0 = 0.455

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


    def pixelToWorld(self, image, u, v, x0, y0, annotateImage=True):
        '''
        Convert the (u,v) pixel position into (x,y) world coordinates
        Inputs:
          image: The image as seen by the camera
          u:     The horizontal (column) pixel coordinate
          v:     The vertical (row) pixel coordinate
          x0:    The x world coordinate in the center of the marker paper
          y0:    The y world coordinate in the center of the marker paper
          annotateImage: Annotate the image with the marker information

        Outputs:
          point: The (x,y) world coordinates matching (u,v), or None

        Return None for the point if not all the Aruco markers are detected
        '''

        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
            image, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        if annotateImage:
            cv2.aruco.drawDetectedMarkers(image, markerCorners, markerIds)

        if (markerIds is None or len(markerIds) != 4 or
            set(markerIds.flatten()) != set([1,2,3,4])):
            return None

        uvMarkers = np.zeros((4,2), dtype='float32')
        for i in range(4):
            uvMarkers[markerIds[i]-1,:] = np.mean(markerCorners[i], axis=1)

        DX = 0.1016
        DY = 0.06985
        xyMarkers = np.float32([[x0+dx, y0+dy] for (dx, dy) in
                                [(-DX, DY), (DX, DY), (-DX, -DY), (DX, -DY)]])

        M = cv2.getPerspectiveTransform(uvMarkers, xyMarkers)

        uvObj = np.float32([u, v])
        xyObj = cv2.perspectiveTransform(uvObj.reshape(1,1,2), M).reshape(2)

        if annotateImage:
            s = "(%7.4f, %7.4f)" % (xyObj[0], xyObj[1])
            cv2.putText(image, s, (u-80, v-8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2, cv2.LINE_AA)

        return xyObj


    def process(self, msg):
        self.object_array.objects = []

        assert(msg.encoding == "rgb8")
        frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")

        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

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

                        strip_world = self.pixelToWorld(frame, int(um), int(vm), self.x0, self.y0, annotateImage=False)
                        if strip_world is not None:
                            strip_world_center_x, strip_world_center_y = strip_world
                            obj_rect = Object()
                            obj_rect.type = Object.STRIP
                            obj_rect.x = float(strip_world_center_x)
                            obj_rect.y = float(strip_world_center_y)
                            obj_rect.z = 0.0
                            obj_rect.theta = angle

                            self.object_array.objects.append(obj_rect)
                        else:
                            self.get_logger().info("PANICCCC!!!! strip_world is None")

                    else:
                        cv2.ellipse(frame, ellipse, self.green, 2)
                        cv2.circle(frame, (int(ue), int(ve)), 5, self.red,    -1)

                        disk_world = self.pixelToWorld(frame, int(ue), int(ve), self.x0, self.y0, annotateImage=False)
                        if disk_world is not None:
                            disk_world_x, disk_world_y = disk_world
                            obj_disk = Object()
                            obj_disk.type = Object.DISK
                            obj_disk.x = float(disk_world_x)
                            obj_disk.y = float(disk_world_y)
                            obj_disk.z = 0.0
                            obj_disk.theta = 0.0

                            self.object_array.objects.append(obj_disk)
                        else:
                            self.get_logger().info("PANICCCC!!!! disk_world is None")

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