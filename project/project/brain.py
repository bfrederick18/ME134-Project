import cv2
import numpy as np

import rclpy
import cv_bridge

from rclpy.node import Node
from sensor_msgs.msg import Image

from geometry_msgs.msg import Point
from project_msgs.msg import Object, ObjectArray, PointArray


class DemoNode(Node):
    red    = (255,   0,   0)
    green  = (  0, 255,   0)
    blue   = (  0,   0, 255)
    yellow = (255, 255,   0)
    white  = (255, 255, 255)

    def __init__(self, name):
        super().__init__(name)

        # Assume the center of marker sheet is at the world origin.
        self.x0 = 0.664
        self.y0 = 0.455

        self.object_array = ObjectArray()
        self.object_array.objects = []

        self.point_array = PointArray()
        self.point_array.points = []

        self.pubrgb = self.create_publisher(Image, name+'/image_raw', 3)

        self.pub_points = self.create_publisher(PointArray, name + '/points_array', 1)
        
        self.get_logger().info('Name: %s' % name)

        self.bridge = cv_bridge.CvBridge()

        self.sub = self.create_subscription(
            Image, '/image_raw', self.process, 1)
        
        self.sub_obj_array = self.create_subscription(
            ObjectArray, '/detector/object_array', self.recv_obj_array, 1)
        
        self.get_logger().info("Detector running...")


    def shutdown(self):
        self.destroy_node()


    def recv_obj_array(self, msg):
        self.object_array.objects = []

        for obj in msg.objects:
            self.object_array.objects.append(obj)


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
        assert(msg.encoding == "rgb8")

        image = self.bridge.imgmsg_to_cv2(msg, "passthrough")

        (H, W, D) = image.shape
        uc = W//2
        vc = H//2

        xyCenter = self.pixelToWorld(image, uc, vc, self.x0, self.y0, annotateImage=False)
        cv2.circle(image, (uc, vc), 5, self.red, -1)

        if xyCenter is None:
            pass
        else:
            (xc, yc) = xyCenter
            # self.get_logger().info("Camera pointed at (%f,%f)" % (xc, yc))

        for obj in self.object_array.objects:
            if obj.type == Object.DISK:
                disc_world = self.pixelToWorld(image, int(obj.x), int(obj.y), self.x0, self.y0, annotateImage=False)
                
                if disc_world is not None:
                    disc_world_msg = Point()
                    disc_world_x, disc_world_y = disc_world
                    disc_world_msg.x = float(disc_world_x)
                    disc_world_msg.y = float(disc_world_y)
                    disc_world_msg.z = 0.0
                    self.point_array.points.append(disc_world_msg)
            
            elif obj.type == Object.STRIP:
                strip_world_center = self.pixelToWorld(image, int(obj.x), int(obj.y), self.x0, self.y0, annotateImage=False)
                if strip_world_center is not None:
                    strip_world_start_msg = Point()
                    strip_world_end_msg = Point()

                    TAP_FACTOR = 10

                    strip_world_start_x = obj.x - TAP_FACTOR * np.sin(np.radians(obj.theta))
                    strip_world_start_y = obj.y + TAP_FACTOR * np.cos(np.radians(obj.theta))
                    strip_world_end_x = obj.x + TAP_FACTOR * np.sin(np.radians(obj.theta))
                    strip_word_end_y = obj.y - TAP_FACTOR * np.cos(np.radians(obj.theta))

                    strip_world_start_msg.x = strip_world_start_x
                    strip_world_start_msg.y = strip_world_start_y
                    strip_world_end_msg.x = strip_world_end_x
                    strip_world_end_msg.y = strip_word_end_y

                    self.point_array.points.append(strip_world_start_msg)
                    self.point_array.points.append(strip_world_end_msg)

        if len(self.point_array.points) > 0:
            self.pub_points.publish(self.point_array)
            self.get_logger().info('All points: %s' % self.point_array.points)

            self.point_array.points = []

        self.pubrgb.publish(self.bridge.cv2_to_imgmsg(image, "rgb8"))


def main(args=None):
    rclpy.init(args=args)
    node = DemoNode('brain')
    rclpy.spin(node)
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()