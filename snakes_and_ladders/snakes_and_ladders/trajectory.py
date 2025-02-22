import numpy as np
import rclpy
from enum import Enum
from math import sin, cos, pi, dist

from rclpy.node         import Node
from sensor_msgs.msg    import JointState

from project_msgs.msg import PointArray, Segment, SegmentArray, State

from hw5code.TrajectoryUtils import goto, spline, goto5, spline5
from hw6sols.KinematicChainSol import KinematicChain
from snakes_and_ladders.constants import CYCLE, JOINT_NAMES, LOGGER_LEVEL, TRAJ_RATE, WAITING_POS


class Mode(Enum):
    WAITING = 0
    POINTING = 1
    RETURNING = 2
    START_UP = 3
    POINTING_RECT_START = 4
    POINTING_RECT_MID = 5
    POINTING_RECT_END = 6


class Spline():
    # Initialization connecting last command to next segment
    def __init__(self, tcmd, pcmd, vcmd, segment):
        # Save the initial time and duration.
        self.t0 = tcmd
        self.T = segment.t

        # Pre-compute the parameters.
        p0 = np.array(pcmd)
        v0 = np.array(vcmd)
        pf = np.array(segment.p)
        vf = np.array(segment.v)
        T = self.T

        self.a = p0
        self.b = v0
        self.c = np.zeros_like(p0)
        self.d = + 10 * (pf - p0) / T ** 3 - 6 * v0 / T ** 2 - 4 * vf / T ** 2
        self.e = - 15 * (pf - p0) / T ** 4 + 8 * v0 / T ** 3 + 7 * vf / T ** 3
        self.f = + 6 * (pf - p0) / T ** 5 - 3 * v0 / T ** 4 - 3 * vf / T ** 4
    
    # Evaluation at any time (Shortening self to s).
    def evaluate(s, t):
        # Get the time relative to the start time.
        t = t - s.t0

        # Compute the current commands.
        p = s.a + s.b*t + s.c*t**2 + s.d*t**3 + s.e*t**4 + s.f*t**5
        v = s.b + 2*s.c*t + 3*s.d*t**2 + 4*s.e*t**3 + 5*s.f*t**4
        
        # Return as a list.
        return (p.tolist(),v.tolist())


class DemoNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().set_level(LOGGER_LEVEL)
        self.get_logger().info('Name: %s' % name)

        self.chain = KinematicChain(self, 'world', 'tip', JOINT_NAMES)
        self.mode = Mode.START_UP

        self.position0 = self.grabfbk()
        # self.get_logger().info("Initial positions: %r" % self.position0)

        self.t = 0
        self.t_start = 0
        (ptip, _, _, _) = self.chain.fkin(WAITING_POS)
        #self.get_logger().info("Received a list of segments: %r" % ptip)
        self.x_waiting = ptip
        self.qD = WAITING_POS
        self.xD = ptip
        self.qddot = [0.0, 0.0, 0.0, 0.0]
        self.qgoal = None
        self.lastpointcmd = self.x_waiting
        self.pointcmd = self.x_waiting
        self.actpos = self.position0.copy()

        self.A = -3.3
        self.B = 0
        self.C = -2.8
        self.D = 0

        self.segments = []
        self.spline = None
        self.abort = False
        self.tcmd = 0
        self.pcmd = WAITING_POS[:]
        self.vcmd = [0.0, 0.0, 0.0, 0.0]

        self.cmd_msg = JointState()
        self.cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)

        self.state_pub = self.create_publisher(State, name + '/state', 10)

        self.sub_seg_array = self.create_subscription(
            SegmentArray, '/brain/segment_array', self.recv_segment_array, 1)


        self.get_logger().info("Waiting for a /joint_commands subscriber...")
        while(not self.count_subscribers('/joint_commands')):
            pass

        self.fbksub = self.create_subscription(
            JointState, '/joint_states', self.recvfbk, 10)

        rate           = TRAJ_RATE
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
        self.cmd_msg.header.stamp = self.get_clock().now().to_msg()
        self.cmd_msg.name         = JOINT_NAMES
        self.cmd_msg.position     = pos
        self.cmd_msg.velocity     = vel
        self.cmd_msg.effort       = eff
        self.cmd_pub.publish(self.cmd_msg)


    # Receive feedback - called repeatedly by incoming messages.
    def recvfbk(self, fbkmsg):
        self.actpos = fbkmsg.position

        state = State()
        state.x_waiting_x = self.x_waiting[0]
        state.x_waiting_y = self.x_waiting[1]
        state.x_waiting_z = self.x_waiting[2]

        state.actpos_x = self.actpos[0]
        state.actpos_y = self.actpos[1]
        state.actpos_z = self.actpos[2]
        state.actpos_w = self.actpos[3]

        self.state_pub.publish(state)


    def recv_segment_array(self, msg):
        self.get_logger().info('Received a list of segments: %r' % msg.segments)
        if self.mode is Mode.WAITING:
            self.segments = msg.segments

            self.tcmd = self.t
            self.pcmd = self.actpos[:]
            self.vcmd = [0.0, 0.0, 0.0, 0.0]

            self.set_mode(Mode.POINTING)


    def super_smart_goto(self, t, initial_pos, final_pos, cycle):
        (q, qdot) = goto5(t, cycle, np.array(initial_pos).reshape(4, 1), np.array(final_pos).reshape(4, 1))
        return q.flatten().tolist(), qdot.flatten().tolist()


    def set_mode(self, new_mode):
        self.mode = new_mode
        self.t_start = self.t


    def set_pointcmd(self, new_pointcmd):
        self.lastpointcmd = self.pointcmd
        self.pointcmd = new_pointcmd


    def gravity(self, pos):
        theta_el = pos[2]
        theta_sh = pos[1]
        tau_elbow = self.C * sin(theta_el - theta_sh) + self.D * cos(theta_el - theta_sh)
        tau_sh = -1 * tau_elbow + self.A * sin(theta_sh) + self.B * cos(theta_sh)
        #self.get_logger().info("Shoulder Torque: %r" % tau_sh)
        return [0.0, tau_sh, tau_elbow, 0.0]
    

    def update(self):
        now = self.get_clock().now()
        self.t  = (now - self.starttime).nanoseconds * 1e-9

        if self.mode is Mode.START_UP:
            if self.t < CYCLE:
                qd, qddot = self.super_smart_goto(self.t, self.position0, [self.position0[0], WAITING_POS[1], self.position0[2], self.position0[3]], CYCLE)
            elif self.t < 2 * CYCLE:
                qd, qddot = self.super_smart_goto(self.t - CYCLE, [self.position0[0], WAITING_POS[1], self.position0[2], self.position0[3]], 
                                                  [WAITING_POS[0], WAITING_POS[1], self.position0[2], self.position0[3]], CYCLE)
            elif self.t < 3 * CYCLE:
                qd, qddot = self.super_smart_goto(self.t - CYCLE * 2, [WAITING_POS[0], WAITING_POS[1], self.position0[2], self.position0[3]], 
                                                  [WAITING_POS[0], WAITING_POS[1], WAITING_POS[2], self.position0[3]], CYCLE)
            elif self.t < 4 * CYCLE:
                qd, qddot = self.super_smart_goto(self.t - CYCLE * 3, [WAITING_POS[0], WAITING_POS[1], WAITING_POS[2], 
                                                                       self.position0[3]], WAITING_POS, CYCLE)
            else:
                qd, qddot = WAITING_POS, [0.0, 0.0, 0.0, 0.0]
                self.get_logger().info('WAITING')
                self.set_mode(Mode.WAITING)

        elif self.mode is Mode.POINTING:
            if self.spline and ((self.t - self.spline.t0) > self.spline.T or self.abort):
                self.spline = None
                self.abort = False
                self.tcmd = self.t

            if not self.spline and len(self.segments) > 0:
                next_seg = self.segments.pop(0)
                self.spline = Spline(self.tcmd, self.pcmd, self.vcmd, next_seg)

            if self.spline:
                (self.pcmd, self.vcmd) = self.spline.evaluate(self.t)
                qd = self.pcmd
                qddot = self.vcmd
            else:
                qd, qddot = self.pcmd, [0.0, 0.0, 0.0, 0.0]

            if self.spline is None and len(self.segments) == 0:
                self.get_logger().info("Trajectory complete, switching to WAITING mode.")
                self.set_mode(Mode.WAITING)
                self.pointcmd = self.x_waiting
                qd, qddot = WAITING_POS, [0.0, 0.0, 0.0, 0.0]
            
            if abs(dist(self.actpos, qd)) > 0.1:
                self.spline = None

                a_seg = Segment()
                a_seg.p = WAITING_POS
                a_seg.v = [0.0, 0.0, 0.0, 0.0]
                a_seg.t = CYCLE
                self.segments = [a_seg]

                self.tcmd = (now - self.starttime).nanoseconds * 1e-9
                self.pcmd = self.actpos[:]
                self.vcmd = [0.0, 0.0, 0.0, 0.0]

                qd = self.qD
                qddot = self.qddot

                self.get_logger().info("HIT RETURNING: %s" % (self.mode))

        else: 
            qd, qddot = WAITING_POS, [0.0, 0.0, 0.0, 0.0]
        
        tau = self.gravity(self.actpos)
        self.sendcmd(qd, qddot, tau)


def main(args=None):
    rclpy.init(args=args)
    node = DemoNode('trajectory')

    rclpy.spin(node)

    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()