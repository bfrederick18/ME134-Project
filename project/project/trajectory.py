import numpy as np
import rclpy
from enum import Enum
from math import sin, cos, pi, dist

from rclpy.node         import Node
from sensor_msgs.msg    import JointState
from geometry_msgs.msg  import Point, Pose

from project_msgs.msg import PointArray

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


class Spline():
    # Initialization connecting last command to next segment
    def __init__(self, tcmd, pcmd, vcmd, segment):
        # Save the initial time and duration.
        self.t0 = tcmd
        self.T = segment.Tmove

        # Pre-compute the parameters.
        p0 = np.array(pcmd)
        v0 = np.array(vcmd)
        pf = np.array(segment.pf)
        vf = np.array(segment.vf)
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


class Segment:
    def __init__(self, pf, vf, Tmove):
        self.pf = pf      # final joint positions (list)
        self.vf = vf      # final joint velocities (list)
        self.Tmove = Tmove  # movement time for this segment (seconds)


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
        self.qddot = [0.0, 0.0, 0.0]
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
        
        self.point_array = PointArray()
        self.point_array.points = []

        self.sub_point_array = self.create_subscription(
            PointArray, '/brain/points_array', self.recvpoint_list, 1)
        
        self.segments = []
        self.spline = None
        self.abort = False
        self.tcmd = 0
        self.pcmd = WAITING_POS[:]
        self.vcmd = [0.0, 0.0, 0.0]

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


    def recvpoint_list(self, msg):
        self.get_logger().info("Received a list of points: %r" % msg.points)
        if self.mode is Mode.WAITING:
            self.point_array.points = msg.points[:]

            if len(self.point_array.points) > 0:
                first_point = self.point_array.points[0]
                if ((first_point.x - 0.7455) ** 2 + (first_point.y - 0.04) ** 2 + (first_point.z - 0.11) ** 2) ** 0.5 < 0.74 and first_point.z >= 0.0 and first_point.y > 0.0:
                    self.set_pointcmd([first_point.x, first_point.y, first_point.z])
                    self.set_mode(Mode.POINTING)
                else:
                    self.get_logger().info('First point not in safety dome. Ignoring...')


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
        tau_shoulder = self.A * sin(theta_sh) + self.B * cos(theta_sh) - 0.1
        return [0.0, tau_shoulder, 0.0]
    
    
    def newton_raphson(self, xgoal):
        xdistance = []
        qstepsize = []
        q = self.actpos
        N = 500

        for i in range(N+1):
            (x, _, Jv, _) = self.chain.fkin(q)
            xdelta = (xgoal - x)
            qdelta = np.linalg.inv(Jv) @ xdelta
            q = q + qdelta * 0.5
            xdistance.append(np.linalg.norm(xdelta))
            qstepsize.append(np.linalg.norm(qdelta))

            if np.linalg.norm(x-xgoal) < 1e-12:
                self.get_logger().info("Completed in %d iterations" % i)
                return q.tolist()
            
        return WAITING_POS


    def update(self):
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
            if len(self.segments) == 0 and len(self.point_array.points) > 0:
                # populate self.segments

                cart_points = [self.x_waiting]
                for pt in self.point_array.points:
                    cart_points.append([pt.x, pt.y, pt.z])
                self.point_array.points = []
                
                Tmove = CYCLE / 2

                for i in range(len(cart_points) - 1):
                    p1 = cart_points[i]
                    p2 = cart_points[i + 1]

                    transitional = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, 0.07]

                    qT = self.newton_raphson(transitional)
                    q2 = self.newton_raphson(p2)

                    if i == 0:
                        self.segments.append(Segment(pf=q2, vf=[0.0, 0.0, 0.0], Tmove=Tmove*2))
                        continue

                    dx = (transitional[0] - p1[0])
                    dy = (transitional[1] - p1[1])
                    v_cart = np.array([dx / Tmove, dy / Tmove, 0.0])

                    (_, _, Jv, _) = self.chain.fkin(qT)
                    qdotT = np.linalg.pinv(Jv) @ v_cart
                    qdotT = qdotT.flatten().tolist()

                    seg1 = Segment(pf=qT, vf=qdotT, Tmove=Tmove)
                    seg2 = Segment(pf=q2, vf=[0.0, 0.0, 0.0], Tmove=Tmove)

                    self.segments.append(seg1)
                    self.segments.append(seg2)

                self.segments.append(Segment(pf=WAITING_POS, vf=[0.0, 0.0, 0.0], Tmove=Tmove*2))

                self.tcmd = (now - self.starttime).nanoseconds * 1e-9  # self.t
                self.pcmd = self.actpos[:]  # use current joint state
                self.vcmd = [0.0, 0.0, 0.0]

                qd = self.qD
                qddot = self.qddot

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
                qd, qddot = self.pcmd, [0.0, 0.0, 0.0]

            if self.spline is None and len(self.segments) == 0 and len(self.point_array.points) == 0:
                self.get_logger().info("Trajectory complete, switching to WAITING mode.")
                self.set_mode(Mode.WAITING)
                self.pointcmd = self.x_waiting
                qd, qddot = WAITING_POS, [0.0, 0.0, 0.0]
            
            if abs(dist(self.actpos, qd)) > 0.05:
                self.spline = None

                self.segments = [Segment(pf=WAITING_POS, vf=[0.0, 0.0, 0.0], Tmove=CYCLE)]
                self.tcmd = (now - self.starttime).nanoseconds * 1e-9
                self.pcmd = self.actpos[:]
                self.vcmd = [0.0, 0.0, 0.0]

                qd = self.qD
                qddot = self.qddot

                self.get_logger().info("HIT RETURNING: %s" % (self.mode))

        else: 
            qd, qddot = WAITING_POS, [0.0, 0.0, 0.0]
        
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