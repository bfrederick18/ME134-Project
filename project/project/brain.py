import cv2
import numpy as np

import rclpy
import cv_bridge

from rclpy.node         import Node
from sensor_msgs.msg    import Image

from geometry_msgs.msg  import Point, Pose


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

        self.pubrgb = self.create_publisher(Image, name+'/image_raw', 3)

        self.pub_point = self.create_publisher(Point, name + '/disc_world', 1)
        self.pub_point_start = self.create_publisher(Point, name + '/rect_start', 1)
        self.pub_point_end = self.create_publisher(Point, name + '/rect_end', 1)
        self.pub_point_center = self.create_publisher(Point, name + '/rect_center', 1)

        self.get_logger().info('Name: %s' % name)

        self.bridge = cv_bridge.CvBridge()

        self.sub = self.create_subscription(
            Image, '/image_raw', self.process, 1)
        
        self.disc_location = [0.0, 0.0]
        self.discsub = self.create_subscription(
            Point, '/balldetector/disc_location', self.recv_disc_location, 1)
        
        self.rect_location_start = [0.0, 0.0]
        self.rect_location_end = [0.0, 0.0]
        self.rect_location = [0.0, 0.0]
        self.rectsubstart = self.create_subscription(
            Point, '/balldetector/strip_pose_start', self.recv_rect_location_start, 1)
        self.rectsubend = self.create_subscription(
            Point, '/balldetector/strip_pose_end', self.recv_rect_location_end, 1)
        self.rectsubcenter = self.create_subscription(
            Point, '/balldetector/strip_pose_center', self.recv_rect_location_center, 1)
        # Report.
        self.get_logger().info("Mapper running...")


    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.destroy_node()


    def recv_disc_location(self, pointmsg):
        x = pointmsg.x
        y = pointmsg.y
        self.disc_location = [x, y]

        # self.get_logger().info('Disc Location: %r' % self.disc_location)
    

    def recv_rect_location_start(self, rect_point_start_msg):
        x = rect_point_start_msg.x
        y = rect_point_start_msg.y
        self.rect_location_start = [x, y]


    def recv_rect_location_end(self, rect_point_end_msg):
        x = rect_point_end_msg.x
        y = rect_point_end_msg.y
        self.rect_location_end = [x, y]


    def recv_rect_location_center(self, rect_point_center_msg):
        x = rect_point_center_msg.x
        y = rect_point_center_msg.y
        self.rect_location = [x, y]

    # Pixel Conversion
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

        # Detect the Aruco markers (using the 4X4 dictionary).
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
            image, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        if annotateImage:
            cv2.aruco.drawDetectedMarkers(image, markerCorners, markerIds)

        # Abort if not all markers are detected.
        if (markerIds is None or len(markerIds) != 4 or
            set(markerIds.flatten()) != set([1,2,3,4])):
            return None


        # Determine the center of the marker pixel coordinates.
        uvMarkers = np.zeros((4,2), dtype='float32')
        for i in range(4):
            uvMarkers[markerIds[i]-1,:] = np.mean(markerCorners[i], axis=1)

        # Calculate the matching World coordinates of the 4 Aruco markers.
        DX = 0.1016
        DY = 0.06985
        xyMarkers = np.float32([[x0+dx, y0+dy] for (dx, dy) in
                                [(-DX, DY), (DX, DY), (-DX, -DY), (DX, -DY)]])


        # Create the perspective transform.
        M = cv2.getPerspectiveTransform(uvMarkers, xyMarkers)

        # Map the object in question.
        uvObj = np.float32([u, v])
        xyObj = cv2.perspectiveTransform(uvObj.reshape(1,1,2), M).reshape(2)


        # Mark the detected coordinates.
        if annotateImage:
            # cv2.circle(image, (u, v), 5, (0, 0, 0), -1)
            s = "(%7.4f, %7.4f)" % (xyObj[0], xyObj[1])
            cv2.putText(image, s, (u-80, v-8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2, cv2.LINE_AA)

        return xyObj


    # Process the image (detect the ball).
    def process(self, msg):
        # Confirm the encoding and report.
        assert(msg.encoding == "rgb8")
        # self.get_logger().info(
        #     "Image %dx%d, bytes/pixel %d, encoding %s" %
        #     (msg.width, msg.height, msg.step/msg.width, msg.encoding))

        # Convert into OpenCV image, using RGB 8-bit (pass-through).
        image = self.bridge.imgmsg_to_cv2(msg, "passthrough")

        # Grab the image shape, determine the center pixel.
        (H, W, D) = image.shape
        uc = W//2
        vc = H//2

        # Convert the center of the image into world coordinates.
        # self.get_logger().info('Camera Cordinates: %s, %s' % (uc, vc))
        xyCenter = self.pixelToWorld(image, uc, vc, self.x0, self.y0, annotateImage=False)
        
        # self.get_logger().info('Types: %s, %s, %s, %s' % (uc, vc, int(self.disc_location[0]), int(self.disc_location[1])))
        # self.get_logger().info('Types: %s, %s, %s, %s' % (type(uc), type(vc), type(self.disc_location[0]), type(self.disc_location[1])))

        if self.disc_location[0] is not 0.0 and self.disc_location[1] is not 0.0:
            disc_world = self.pixelToWorld(image, int(self.disc_location[0]), int(self.disc_location[1]), self.x0, self.y0, annotateImage=False)
            if disc_world is not None:
                # self.get_logger().info('World Disc Location: %s' % disc_world)
                disc_world_msg = Point()
                disc_world_x, disc_world_y = disc_world
                disc_world_msg.x = float(disc_world_x)
                disc_world_msg.y = float(disc_world_y)
                disc_world_msg.z = 0.0
                self.pub_point.publish(disc_world_msg)

            self.disc_location = [0.0, 0.0]

        if self.rect_location[0] is not 0.0 and self.rect_location[1] is not 0.0:
            rect_world = self.pixelToWorld(image, int(self.rect_location[0]), int(self.rect_location[1]), self.x0, self.y0, annotateImage=False)
            rect_world_start = self.pixelToWorld(image, int(self.rect_location_start[0]), int(self.rect_location_start[1]), self.x0, self.y0, annotateImage=False)
            rect_world_end = self.pixelToWorld(image, int(self.rect_location_end[0]), int(self.rect_location_end[1]), self.x0, self.y0, annotateImage=False)
            if rect_world is not None:
                # self.get_logger().info('Start: %s, End: %s' % (self.rect_location_start, self.rect_location_end))

                rect_world_msg_center = Point()
                rect_world_x_center, rect_world_y_center = rect_world
                rect_world_msg_center.x = float(rect_world_x_center)
                rect_world_msg_center.y = float(rect_world_y_center)
                rect_world_msg_center.z = 0.0
                self.pub_point_center.publish(rect_world_msg_center)

                rect_world_msg_start = Point()
                rect_world_x_start, rect_world_y_start = rect_world_start
                rect_world_msg_start.x = float(rect_world_x_start)
                rect_world_msg_start.y = float(rect_world_y_start)
                rect_world_msg_start.z = 0.0
                self.pub_point_start.publish(rect_world_msg_start)

                rect_world_msg_end = Point()
                rect_world_x_end, rect_world_y_end = rect_world_end
                rect_world_msg_end.x = float(rect_world_x_end)
                rect_world_msg_end.y = float(rect_world_y_end)
                rect_world_msg_end.z = 0.0
                self.pub_point_end.publish(rect_world_msg_end)

            self.rect_location_start = [0.0, 0.0]
            self.rect_location_end = [0.0, 0.0]
            self.rect_location = [0.0, 0.0]

        # Mark the center of the image.
        cv2.circle(image, (uc, vc), 5, self.red, -1)

        # Report the mapping.
        if xyCenter is None:
            # self.get_logger().info("Unable to execute mapping")
            pass
        else:
            (xc, yc) = xyCenter
            # self.get_logger().info("Camera pointed at (%f,%f)" % (xc, yc))

        # Convert the image back into a ROS image and republish.
        self.pubrgb.publish(self.bridge.cv2_to_imgmsg(image, "rgb8"))


#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the detector node.
    node = DemoNode('brain')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()