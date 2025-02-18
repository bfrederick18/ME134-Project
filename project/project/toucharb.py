import numpy as np
import rclpy
from enum import Enum
from math import sin, cos, pi

from rclpy.node         import Node
from sensor_msgs.msg    import JointState
from geometry_msgs.msg  import Point, Pose

from hw5code.TrajectoryUtils import goto, spline, goto5, spline5
from hw6sols.KinematicChainSol import KinematicChain


RATE = 100.0  # Hertz
CYCLE = 3 * pi / 2
WAITING_POS = [0.0, 0.0, -pi / 2]
JOINT_NAMES = ['base', 'shoulder', 'elbow']


class Mode(Enum):
    WAITING = 0
    POINTING = 1
    RETURNING = 2
    START_UP = 3
    POINTING_RECT_START = 4
    POINTING_RECT_MID = 5
    POINTING_RECT_END = 6


class DemoNode(Node):
    def __init__(self, name):
        super().__init__(name)

        self.chain = KinematicChain(self, 'world', 'tip', JOINT_NAMES)
        self.mode = Mode.START_UP

        self.position0 = self.grabfbk()
        # self.get_logger().info("Initial positions: %r" % self.position0)

        self.t = 0
        self.t_start = 0
        (ptip, _, _, _) = self.chain.fkin(WAITING_POS)
        self.x_waiting = ptip
        self.qD = WAITING_POS
        self.xD = ptip
        self.qddot = None
        self.qgoal = None
        self.lastpointcmd = self.x_waiting
        self.pointcmd = self.x_waiting
        self.actpos = self.position0.copy()
        self.A = -1.85
        self.B = 0

        self.cmdmsg = JointState()
        self.cmdpub = self.create_publisher(JointState, '/joint_commands', 10)

        self.get_logger().info("Waiting for a /joint_commands subscriber...")
        while(not self.count_subscribers('/joint_commands')):
            pass

        self.fbksub = self.create_subscription(
            JointState, '/joint_states', self.recvfbk, 10)
        
        self.pointsub = self.create_subscription(
            Point, '/point', self.recvpoint, 10)

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
        self.cmdmsg.name         = JOINT_NAMES
        self.cmdmsg.position     = pos
        self.cmdmsg.velocity     = vel
        self.cmdmsg.effort       = eff
        self.cmdpub.publish(self.cmdmsg)


    # Receive feedback - called repeatedly by incoming messages.
    def recvfbk(self, fbkmsg):
        self.actpos = fbkmsg.position
        pass


    def recvpoint(self, pointmsg):
        x = pointmsg.x
        y = pointmsg.y
        z = pointmsg.z
        
        if self.mode is Mode.WAITING:
            if ((x - 0.7455) ** 2 + (y - 0.04) ** 2 + (z - 0.11) ** 2) ** (1 / 2) < 0.74 and z >= 0.0 and y > 0.0:
                self.set_pointcmd([x, y, z])
                self.qgoal = self.newton_raphson(self.pointcmd)
                self.get_logger().info("qgoal: %r" % self.qgoal)
                self.set_mode(Mode.POINTING)
            
                self.get_logger().info("Running point %r, %r, %r" % (x,y,z))
            else:
                self.get_logger().info('Not in the dome. Please try again dummy...')
        else:
            self.get_logger().info('Not waiting yet. Please wait dummy...')


    def super_smart_goto(self, t, initial_pos, final_pos, cycle):
        (q, qdot) = goto5(t, cycle, np.array(initial_pos).reshape(3, 1), np.array(final_pos).reshape(3, 1))
        return q.flatten().tolist(), qdot.flatten().tolist()


    def set_mode(self, new_mode):
        self.mode = new_mode
        self.t_start = self.t


    def set_pointcmd(self, new_pointcmd):
        self.lastpointcmd = self.pointcmd
        self.pointcmd = new_pointcmd


    def gravity(self, pos):
        theta_sh = pos[1]
        tau_shoulder = self.A * sin(theta_sh) + self.B * cos(theta_sh) 
        return [0.0, tau_shoulder, 0.0]
    
    
    def newton_raphson(self, xgoal):
        xdistance = []
        qstepsize = []
        q = self.actpos
        N = 1000
        for i in range(N+1):
            (x, _, Jv, _) = self.chain.fkin(q)
            xdelta = (xgoal - x)
            qdelta = np.linalg.inv(Jv) @ xdelta
            q = q + qdelta * 0.3
            xdistance.append(np.linalg.norm(xdelta))
            qstepsize.append(np.linalg.norm(qdelta))

            if np.linalg.norm(x-xgoal) < 1e-12:
                self.get_logger().info("Completed in %d iterations" % i)
                return q.tolist()
        return WAITING_POS


    def update(self):
        # Grab the current time.
        now = self.get_clock().now()
        self.t  = (now - self.starttime).nanoseconds * 1e-9

        if self.mode is Mode.START_UP:
            if self.t < CYCLE:
                qd, qddot = self.super_smart_goto(self.t, self.position0, [self.position0[0], WAITING_POS[1], self.position0[2]], CYCLE)
            elif self.t < 2 * CYCLE:
                qd, qddot = self.super_smart_goto(self.t - CYCLE, [self.position0[0], WAITING_POS[1], self.position0[2]], [WAITING_POS[0], WAITING_POS[1], self.position0[2]], CYCLE)
            elif self.t < 3 * CYCLE:
                qd, qddot = self.super_smart_goto(self.t - CYCLE * 2, [WAITING_POS[0], WAITING_POS[1], self.position0[2]], WAITING_POS, CYCLE)
            else:
                qd, qddot = WAITING_POS, [0.0, 0.0, 0.0]

                self.set_mode(Mode.WAITING)

        elif self.mode is Mode.POINTING:
            if self.t - self.t_start < CYCLE:
                qd, qddot = self.super_smart_goto(self.t - self.t_start, WAITING_POS, self.qgoal, CYCLE)
                self.qD = qd
                self.qddot = qddot

            else:
                qd = self.qD
                qddot = self.qddot

                self.set_mode(Mode.RETURNING)

        elif self.mode is Mode.RETURNING:
            if self.t - self.t_start < CYCLE:
                qd, qddot = self.super_smart_goto(self.t - self.t_start, self.qD, WAITING_POS, CYCLE)
            else:
                qd, qddot = WAITING_POS, [0.0, 0.0, 0.0]
                self.pointcmd = self.x_waiting
                self.set_mode(Mode.WAITING)
                self.get_logger().info("HIT WAITING: %s" % (self.mode))

                self.qD = WAITING_POS
                self.xD = self.x_waiting

        else:  # elif self.mode is Mode.WAITING:
            qd, qddot = WAITING_POS, [0.0, 0.0, 0.0]
        
        tau = self.gravity(self.actpos)
        self.sendcmd(qd, qddot, tau)


def main(args=None):
    rclpy.init(args=args)
    node = DemoNode('toucharb')

    rclpy.spin(node)

    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()