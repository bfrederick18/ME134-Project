import numpy as np
import rclpy
from math import sin, cos, pi

from rclpy.node         import Node
from sensor_msgs.msg    import JointState


RATE = 100.0  # Hertz


class DemoNode(Node):
    def __init__(self, name):
        super().__init__(name)

        self.position0 = self.grabfbk()
        self.get_logger().info("Initial positions: %r" % self.position0)

        self.cmdmsg = JointState()
        self.cmdpub = self.create_publisher(JointState, '/joint_commands', 10)

        self.get_logger().info("Waiting for a /joint_commands subscriber...")
        while(not self.count_subscribers('/joint_commands')):
            pass

        self.fbksub = self.create_subscription(
            JointState, '/joint_states', self.recvfbk, 10)

        rate           = RATE
        self.starttime = self.get_clock().now()
        self.timer     = self.create_timer(1/rate, self.update)
        self.get_logger().info("Sending commands with dt of %f seconds (%fHz)" %
                               (self.timer.timer_period_ns * 1e-9, rate))

    def shutdown(self):
        self.destroy_node()


    # Grab a single feedback - DO NOT CALL THIS REPEATEDLY!
    def grabfbk(self):
        def cb(fbkmsg):
            self.grabpos   = list(fbkmsg.position)
            self.grabready = True

        sub = self.create_subscription(JointState, '/joint_states', cb, 1)
        self.grabready = False
        while not self.grabready:
            rclpy.spin_once(self)
        self.destroy_subscription(sub)

        return self.grabpos


    def sendcmd(self, pos, vel, eff = []):
        self.cmdmsg.header.stamp = self.get_clock().now().to_msg()
        self.cmdmsg.name         = ['one', 'two', 'three']
        self.cmdmsg.position     = pos
        self.cmdmsg.velocity     = vel
        self.cmdmsg.effort       = eff
        self.cmdpub.publish(self.cmdmsg)


    # Receive feedback - called repeatedly by incoming messages.
    def recvfbk(self, fbkmsg):
        # print(list(fbkmsg.position))
        pass


    def update(self):
        # Grab the current time.
        now = self.get_clock().now()
        t   = (now - self.starttime).nanoseconds * 1e-9

        # Time it takes to initialize position for start of wave motion
        START_SHIFT = pi

        # Define range of motor 3
        JOINT_THREE_MAX = 0.5
        JOINT_THREE_MIN = -0.5
        joint_three_amp = JOINT_THREE_MAX - JOINT_THREE_MIN
        
        # Define range of motor 2
        JOINT_TWO_MAX = 0.3
        JOINT_TWO_MIN = -0.3
        JOINT_TWO_SHIFT = -pi / 2
        joint_two_amp = JOINT_TWO_MAX - JOINT_TWO_MIN

        # Define range of motor 1
        JOINT_ONE_MAX = 0.2
        JOINT_ONE_MIN = -0.2
        joint_one_amp = JOINT_ONE_MAX - JOINT_ONE_MIN

        # Initialize motors to position arm at starting position
        if t < START_SHIFT:
            qd    = [(0 - self.position0[0]) * (cos(t + pi) - 1) / 2, 
                     (self.position0[1] - JOINT_TWO_MIN * 2) / 2 * cos(t) + ((self.position0[1] + JOINT_TWO_MIN * 2) / 2), 
                     (0 - self.position0[2]) * (cos(t + pi) - 1) / 2]
            qddot = [0.0, 0.0, 0.0]
        
        # Wave continuously 
        else:
            qd    = [joint_one_amp * cos(t - START_SHIFT + pi / 2), 
                     joint_two_amp * sin(t + JOINT_TWO_SHIFT - START_SHIFT), 
                     joint_three_amp * sin(t - START_SHIFT)]
            qddot = [0.0, 0.0, 0.0]
        
        self.sendcmd(qd, qddot)


def main(args=None):
    rclpy.init(args=args)
    node = DemoNode('wave')

    rclpy.spin(node)

    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()