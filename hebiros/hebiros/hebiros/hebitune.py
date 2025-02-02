#!/usr/bin/env python3
#
#   hebitune.py motorname speed period
#
#   Tune the specified motor.  The command line arguments determine
#   the triangle wave (speed and period) to use.
#
#   This sets up the /joint_states and /joint_commands topics to
#   observe the actual and command signals.  It also listens to /kp,
#   /kv, /lp to set the position gain, velocity gain, and low pass
#   filter.
#
#   Node:           /hebitune
#   Subscribers:    /kp             Float64     Position gain
#                   /kv             Float64     Velocity gain
#                   /lp             Float64     Low pass filter constant
#   Publishers:     /joint_states   JointState  Actual signals
#                   /joint_commands JointState  Command signals
#
import hebi
import numpy as np
import rclpy
import sys

from time import sleep

from rclpy.node         import Node
from sensor_msgs.msg    import JointState
from std_msgs.msg       import Float64


#
#   Definitions
#
RATE     = 100.0        # HEBI feedback rate in Hertz
LIFETIME =  50.0        # HEBI command lifetime in ms

WATCHDOGDT = 0.2        # Watchdog Time step (seconds)

SPEED  = 0.2            # Default triangle wave speed
PERIOD = 5.0            # Default triangle wave period


#
#   HEBI Node Class
#
class HebiNode(Node):
    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Connect to all HEBIs.
        self.hebiconnectall()

        # Pull out the command line parameters, configure the tuning.
        self.configure(sys.argv)

        # Set up the ROS communications and callbacks.
        self.startup()

    # Startup
    def startup(self):
        # Feedback: Create a feedback (actual) and a command message.
        self.fbkmsg      = JointState()
        self.cmdmsg      = JointState()
        self.fbkmsg.name = self.motors          # Set to all motors
        self.cmdmsg.name = self.motors

        # Create a publisher for the /joint_states and /joint_commands
        self.fbkpub = self.create_publisher(JointState, '/joint_states',   10)
        self.cmdpub = self.create_publisher(JointState, '/joint_commands', 10)

        # Create the subscribers
        self.kpsub = self.create_subscription(Float64, '/kp', self.setkpCB, 1)
        self.kvsub = self.create_subscription(Float64, '/kv', self.setkvCB, 1)
        self.lpsub = self.create_subscription(Float64, '/lp', self.setlpCB, 1)

        # Create a HEBI feeedback handler to drive the system.
        self.group.add_feedback_handler(self.feedbackCB)

        # Finally create a watchdog timer to check the HEBI connection.
        self.watchdog = self.create_timer(WATCHDOGDT, self.watchdogCB)

    # Shutdown
    def shutdown(self):
        # Clear the HEBI feedback handler, kill the watchdog timer and node.
        self.group.clear_feedback_handlers()
        self.destroy_timer(self.watchdog)
        self.destroy_node()

    # Info/Warn/Error Messages
    def i(self, str):
        self.get_logger().info(str)
    def w(self, str):
        self.get_logger().warn(str)
    def e(self, str):
        self.get_logger().error(str)



    ##################################################################
    # Locate and connect to all HEBI actuators
    def hebiconnectall(self):
        # Locate HEBI actuators on the network, waiting 1s for discovery.
        self.i("Locating HEBIs...")
        lookup = hebi.Lookup()
        sleep(1)
        if sum(1 for entry in lookup.entrylist) == 0:
            self.e("Unable to locate any HEBI motors!")
            self.e("Make sure motors are powered/connected.")
            self.e("(for example using the Scope application)")
            raise Exception("No HEBIs located")

        # Parse the connected HEBIs and report.
        self.families  = [entry.family      for entry in lookup.entrylist]
        self.motors    = [entry.name        for entry in lookup.entrylist]
        self.addresses = [entry.mac_address for entry in lookup.entrylist]
        for (f,n,a) in zip(self.families, self.motors, self.addresses):
            self.i("Located family '%s' name '%s' at address %s" % (f,n,a))
    
        # Create the HEBI actuator group.
        self.group = lookup.get_group_from_names(self.families, self.motors)
        if self.group is None:
            self.e("Unable to connect to HEBI motors!")
            raise Exception("No HEBI connection")

        # Set the HEBI command lifetime (in ms) and feedback freq (in Hz).
        self.group.command_lifetime   = LIFETIME
        self.group.feedback_frequency = RATE

        # Grab an initial feedback structure.
        self.fbk0 = self.group.get_next_feedback()
        if self.fbk0 is None:
            self.e("Unable to get feedback from HEBI motors!")
            raise Exception("No HEBI feedback")

        # Grab an info structure.
        self.info = self.group.request_info()
        if self.info is None:
            self.e("Unable to get information from HEBI motors!")
            raise Exception("No HEBI information")

        # Allocate/prepare a command structure.
        self.cmd        = hebi.GroupCommand(self.group.size)
        self.cmd.effort = np.full(self.group.size, np.nan)

        # Initialize the commands to the current data.
        self.pos = np.copy(self.fbk0.position)
        self.vel = np.full(self.group.size, 0.0)
        self.kp  = self.info.position_kp
        self.kv  = self.info.velocity_kp
        self.lp  = self.info.velocity_output_lowpass

        # Report.
        self.i("Connected to HEBIs:")
        for i in range(self.group.size):
            str  = "Motor #%d '%s'"         % (i, self.motors[i])
            str += " position %6.3f rad"    % self.fbk0.position[i]
            str += ", Kp %6.3f Nm/rad"      % self.info.position_kp[i]
            str += ", Kv %6.3f PWM/(rad/s)" % self.info.velocity_kp[i]
            str += ", filter %6.3f" % self.info.velocity_output_lowpass[i]
            self.i(str)
            if self.info.position_kd[i] != 0.0:
                self.w("WARNING: Position Kd is not zero!")
            if self.info.position_ki[i] != 0.0:
                self.w("WARNING: Position Ki is not zero!")
            if self.info.velocity_kd[i] != 0.0:
                self.w("WARNING: Velocity Kd is not zero!")
            if self.info.velocity_ki[i] != 0.0:
                self.w("WARNING: Velocity Ki is not zero!")

    ##################################################################
    # Determine the motor to tune (and parameters).
    def configure(self, argv):
        # Check the given parameters
        if (len(argv) < 2) or (len(argv) > 4):
            self.e("Usage: (ros2 run hebiros) hebitune.py name speed period")
            self.e("w/ required name   is the HEBI motor to tune")
            self.e("   optional speed  is the triangle wave speed in rad/s")
            self.e("   optional period is the triangle wave period in s")
            raise Exception("Illegal arguments")

        # Grab and check the motor.
        motor = sys.argv[1]
        if (motor not in self.motors):
            self.e("Motor '%s' not present!" % motor)
            raise Exception("Motor not present")
        self.index = self.motors.index(motor)

        # Grab the triangle wave period
        self.speed = min(0.5*self.info.velocity_max_target[self.index],
                         (0.5*self.info.velocity_max_output[self.index]/
                          self.info.velocity_kp[self.index]))
        self.speed  = SPEED
        self.period = PERIOD
        if (len(argv) > 2):
            self.speed  = float(argv[2])
        if (len(argv) > 3):
            self.period = float(argv[3])

        # Set the upper/lower position bounds
        self.upper = self.fbk0.position[self.index] + self.speed*self.period/4
        self.lower = self.fbk0.position[self.index] - self.speed*self.period/4

        # Initialize the velocity commands to start things off.
        self.vel[self.index] = self.speed

        # Report.
        str  = "Tuning motor #%d '%s'"         % (self.index, motor)
        str += ", position %6.3f to %6.3f rad" % (self.lower, self.upper)
        str += ", speed %5.2f rad/s"           % (self.speed)
        str += ", period %3.1f s"              % (self.period)
        self.i(str)
        self.i("Starting with:")
        self.showgains()

    ##################################################################
    # HEBI feedback callback - send commands and ROS messages.
    def feedbackCB(self, fbk):
        # Grab the current time.
        time = self.get_clock().now()

        # Build up the feedback message and publish (joint names are preset).
        self.fbkmsg.header.stamp = time.to_msg()
        self.fbkmsg.position     = fbk.position.tolist()
        self.fbkmsg.velocity     = fbk.velocity.tolist()
        self.fbkmsg.effort       = fbk.effort.tolist()
        self.fbkpub.publish(self.fbkmsg)

        # Build up the command message and publish (joint names are preset).
        self.cmdmsg.header.stamp = time.to_msg()
        self.cmdmsg.position     = self.cmd.position.tolist()
        self.cmdmsg.velocity     = self.cmd.velocity.tolist()
        self.cmdmsg.effort       = self.cmd.effort.tolist()
        self.cmdpub.publish(self.cmdmsg)

        # Integrate the trajectory (switching at limits)
        self.pos[self.index] += self.vel[self.index] / RATE
        if (self.pos[self.index] > self.upper):
            self.pos[self.index] = 2*self.upper - self.pos[self.index]
            self.vel[self.index] = -self.speed
        elif (self.pos[self.index] < self.lower):
            self.pos[self.index] = 2*self.lower - self.pos[self.index]
            self.vel[self.index] = self.speed

        # Update and send the HEBI commands.
        self.cmd.position                = self.pos
        self.cmd.velocity                = self.vel
        self.cmd.position_kp             = self.kp
        self.cmd.velocity_kp             = self.kv
        self.cmd.velocity_output_lowpass = self.lp
        self.group.send_command(self.cmd)

        # Reset the watchdog.
        self.watchdog.reset()


    ##################################################################
    # Gain Callbacks
    def showgains(self):
        str  = "Motor #%d '%s': "       % (self.index, self.motors[self.index])
        str += "Kp %7.3f Nm/rad, "      % self.kp[self.index]
        str += "Kv %6.3f PWM/(rad/s), " % self.kv[self.index]
        str += "filter %5.3f"           % self.lp[self.index]
        str += " or %4.0f rad/s"        % (self.lp[self.index]*1000)
        str += " or %6.2f Hz"           % (self.lp[self.index]*500/np.pi)
        self.i(str)

    def setkpCB(self, msg):
        self.kp[self.index] = msg.data
        self.showgains()

    def setkvCB(self, msg):
        self.kv[self.index] = msg.data
        self.showgains()

    def setlpCB(self, msg):
        self.lp[self.index] = msg.data
        self.showgains()

    # Watchdog callback
    def watchdogCB(self):
        self.w("Not getting HEBI feedback - check connection")



#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the HEBI node and connect to all HEBIs.
    node = HebiNode('hebitune')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
