#!/usr/bin/env python3
#
#   hebiforce.py motorname amplitude period
#
#   Output only a force sinusoid to the specified motor.  The command
#   line arguments determine the amplitude and period to use.
#
#   This sets up the /joint_states and /joint_commands topics to
#   observe the actual and command signals.
#
#   Node:           /hebiforce
#   Publishers:     /joint_states   JointState  Actual signals
#                   /joint_commands JointState  Command signals
#
import hebi
import numpy as np
import rclpy
import sys

from math import sin, cos, pi
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

AMPLITUDE  = 1.0        # Default force sinusoid amplitude
PERIOD     = 5.0        # Default force sinusoid period


#
#   HEBI Node Class
#
class HebiNode(Node):
    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Lookup/connect to all HEBIs.
        self.hebilookup()
        self.hebiconnectall()

        # Pull out the command line parameters.
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

        # Create a HEBI feeedback handler to drive the system.
        self.time0 = self.get_clock().now()
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
    def hebilookup(self):
        # Locate HEBI actuators on the network, waiting 1s for discovery.
        self.i("Locating HEBIs...")
        self.lookup = hebi.Lookup()
        sleep(1)
        if sum(1 for entry in self.lookup.entrylist) == 0:
            self.e("Unable to locate any HEBI motors!")
            self.e("Make sure motors are powered/connected.")
            self.e("(for example using the Scope application)")
            raise Exception("No HEBIs located")

        # Parse the connected HEBIs and report.
        self.families  = [entry.family      for entry in self.lookup.entrylist]
        self.motors    = [entry.name        for entry in self.lookup.entrylist]
        self.addresses = [entry.mac_address for entry in self.lookup.entrylist]
        for (f,n,a) in zip(self.families, self.motors, self.addresses):
            self.i("Located family '%s' name '%s' at address %s" % (f,n,a))

    def hebiconnectall(self):
        # Create the HEBI actuator group.
        self.group = self.lookup.get_group_from_names(
            self.families, self.motors)
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
        self.cmd          = hebi.GroupCommand(self.group.size)
        self.cmd.position = np.full(self.group.size, np.nan)
        self.cmd.velocity = np.full(self.group.size, np.nan)
        self.cmd.effort   = np.full(self.group.size, np.nan)

        # Report.
        self.i("Connected to HEBIs:")
        for i in range(self.group.size):
            str  = "Motor #%d '%s'"         % (i, self.motors[i])
            str += " position %6.3f rad"    % self.fbk0.position[i]
            str += ", effort %6.3f Nm"      % self.fbk0.effort[i]
            self.i(str)

    ##################################################################
    # Determine the motor to tune (and parameters).
    def configure(self, argv):
        # Check the given parameters
        if (len(argv) < 2) or (len(argv) > 4):
            self.e("Usage: (ros2 run hebiros) hebitune.py name speed period")
            self.e("w/ required name       is the HEBI motor to drive")
            self.e("   optional amplitude  is the force amplitude in Nm")
            self.e("   optional period     is the sinusoid period in s")
            raise Exception("Illegal arguments")

        # Grab and check the motor.
        motor = sys.argv[1]
        if (motor not in self.motors):
            self.e("Motor '%s' not present!" % motor)
            raise Exception("Motor not present")
        self.index = self.motors.index(motor)

        # Grab the sinusoid amplitude and  period.
        self.amplitude = AMPLITUDE if (len(argv) < 3) else float(argv[2])
        self.period    = PERIOD    if (len(argv) < 4) else float(argv[3])

        # Report.
        str  = "Driving motor #%d '%s'" % (self.index, motor)
        str += ", amplitude %5.2f Nm"   % (self.amplitude)
        str += ", period %3.1f s"       % (self.period)
        self.i(str)

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

        # Set the trajectory and send the HEBI commands.
        t = (time - self.time0).nanoseconds * 1e-9
        effort = np.copy(self.cmd.effort)
        effort[self.index] = self.amplitude * sin(2*pi*t/self.period)

        # Update the HEBI commands and send.
        self.cmd.effort = effort
        self.group.send_command(self.cmd)

        # Reset the watchdog.
        self.watchdog.reset()


    ##################################################################
    # ROS Callbacks
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
    node = HebiNode('hebiforce')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
