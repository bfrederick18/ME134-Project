#!/usr/bin/env python3
#
#   hebinode.py
#
#   Create the HEBI node.  This sets up the HEBI connection, then
#   passes then HEBI feedback to the /joint_states topic, while
#   transfering the /joint_commands topic to the HEBIs.
#
#   It maps the HEBI actuator names to joint names according to the
#   ROS parameters:
#
#     family        Name of the HEBI actuator family ('robotlab')
#     motors        List of motor names (['9.3', '9.6', ...])
#     joints        List of joint names (['base', 'joint2', 'eblow', ...])
#     rate          HEBI Feedback rate in Hertz
#     lifetime      HEBI command lifetime in milliseconds
#
#   It also allows to "simulate" the HEBIs if they are not connected
#   or explicitly set by the ROS parameters:
#
#     simulate      True/False whether to return the last cmd as actual.
#
#

try:
    import hebi
    hebiexists = True
except:
    hebiexists = False

import numpy as np
import rclpy

from time      import sleep
from traceback import print_exc

from rclpy.node         import Node
from rclpy.parameter    import Parameter
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import ParameterType

from sensor_msgs.msg    import JointState


#
#   Definitions
#
FAMILY   = 'robotlab'
RATE     = 100.0        # Hertz
LIFETIME =  50.0        # ms

WATCHDOGDT = 0.2        # Watchdog Time step (seconds)


#
#   HEBI Node Class
#
#   This contains the local data:
#
#     self.family       HEBI family name
#     self.motors       List of motor names
#     self.joints       List of joint names
#     self.N            Number of joints/motors
#
#     self.simulate     True/False whether to simulate
#     self.rate         Update rate (100Hz)
#     self.lifetime     Command lifetime (50ms)
#
#     self.fbkmsg.position     List of current joint positions
#     self.fbkmsg.velocity     List of current joint velocities
#     self.fbkmsg.effort       List of current joint efforts
#
class HebiNode(Node):
    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

    # Startup
    def startup(self):
        # Check the ROS parameters.
        self.readparameters()

        # If not simulating, TRY to connect to the HEBIs.
        if not self.simulate:
            # If the setup fails (does not return True), switch to sim!
            if not self.hebisetup():
                self.warn("SWITCHING TO SIMULATION MODE")                
                self.simulate = True

        # Feedback: Create a feedback message (JointState) to contain
        # the current pos/vel/eff and a publisher to send.
        self.fbkmsg              = JointState()
        self.fbkmsg.header.stamp = self.get_clock().now().to_msg()
        self.fbkmsg.name         = self.joints   # Set to the joint names
        self.fbkmsg.position     = [0.0] * self.N
        self.fbkmsg.velocity     = [0.0] * self.N
        self.fbkmsg.effort       = [0.0] * self.N

        self.pub = self.create_publisher(JointState, '/joint_states', 10)

        # Create a watchdog timer to check the HEBI connection.
        self.watchdog = self.create_timer(WATCHDOGDT, self.watchdogCB)

        # Set up the feedback source and handlers based on whether we
        # are simulating or not!
        if self.simulate:
            self.warn("######################################")
            self.warn("######## HEBI SIMULATION MODE ########")
            self.warn("######################################")

            # Feedback from a timer.
            self.fbktimer = self.create_timer(1.0/self.rate, self.fbkCB_sim)

            # Subscribe to the joint commands.
            self.sub  = self.create_subscription(
                JointState, '/joint_commands', self.cmdCB_sim, 10)

        else:
            # Pre-Create a HEBI command structure and a list of NaNs.
            self.cmd  = hebi.GroupCommand(self.group.size)
            self.nans = np.full(self.N, np.nan)

            # Feedback from HEBI group.
            self.group.add_feedback_handler(self.fbkCB_real)

            # Subscribe to the joint commands.
            self.sub  = self.create_subscription(
                JointState, '/joint_commands', self.cmdCB_real, 10)


    # Shutdown
    def shutdown(self):
        # If online, clear the HEBI feedback handler.  Else destroy
        # the feedback timer.  Then kill the watchdog timer and node.
        if self.simulate:
            self.destroy_timer(self.fbktimer)
        else:
            self.group.clear_feedback_handlers()
        self.destroy_timer(self.watchdog)
        self.destroy_node()


    ######################################################################
    # Output Utilities
    def info(self, msg):
        self.get_logger().info(msg)
    def warn(self, msg):
        self.get_logger().warn(msg)
    def error(self, msg):
        self.get_logger().error(msg)


    ######################################################################
    # Grab individual ROS parameters.
    def getbool(self, name, default):
        # Declare the parameter, then get/return the value.
        type       = ParameterType.PARAMETER_BOOL
        descriptor = ParameterDescriptor(type=type)
        self.declare_parameter(name, descriptor=descriptor, value=default)
        parameter  = self.get_parameter(name)
        return parameter.get_parameter_value().bool_value

    def getdouble(self, name, default):
        # Declare the parameter, then get/return the value.
        type       = ParameterType.PARAMETER_DOUBLE
        descriptor = ParameterDescriptor(type=type)
        self.declare_parameter(name, descriptor=descriptor, value=default)
        parameter  = self.get_parameter(name)
        return parameter.get_parameter_value().double_value

    def getstring(self, name, default):
        # Declare the parameter, then get/return the value.
        type       = ParameterType.PARAMETER_STRING
        descriptor = ParameterDescriptor(type=type)
        self.declare_parameter(name, descriptor=descriptor, value=default)
        parameter  = self.get_parameter(name)
        return parameter.get_parameter_value().string_value

    def getstringarray(self, name):
        # Declare the parameter, then get/return the value.
        type       = ParameterType.PARAMETER_STRING_ARRAY
        descriptor = ParameterDescriptor(type=type)
        self.declare_parameter(name, descriptor=descriptor)
        parameter  = self.get_parameter(name)
        return parameter.get_parameter_value().string_array_value

    # Read the ROS parameters.
    def readparameters(self):
        # Get the parameters.
        self.family   = self.getstring('family', FAMILY)
        self.motors   = self.getstringarray('motors')
        self.joints   = self.getstringarray('joints')

        self.rate     = self.getdouble('rate', RATE)
        self.lifetime = self.getdouble('lifetime', LIFETIME)
        self.simulate = self.getbool('simulate', False)

        self.N        = len(self.joints)

        # Check the parameters for consistency.
        if len(self.motors) == 0:
            self.error("No motors specified!")
            raise Exception("Inconsistent ROS parameters")
        if len(self.joints) == 0:
            self.error("No joints specified!")
            raise Exception("Inconsistent ROS parameters")
        if len(self.motors) != len(self.joints):
            self.error("Unequal number of joints/motors specified!")
            raise Exception("Inconsistent ROS parameters")

        # Report the parameters.
        self.info("Reading parameters...")
        self.info("HEBI family  '%s'" % self.family)
        for i in range(self.N):
            self.info("HEBI motor %ld '%s' = joint '%s'" %
                (i, self.motors[i], self.joints[i]))
        self.info("HEBI update rate %3.0fHz" % self.rate)
        self.info("HEBI command lifetime %3.0fms" % self.lifetime)
        self.info("Desired simulation mode %s" % str(self.simulate))


    ######################################################################
    # HEBI Setup.  Switch to simulation mode if this fails!
    def hebisetup(self):
        # Make sure we have the hebi library (only in the NUCs).
        if not hebiexists:
            self.warn("Unable to import HEBI!!!")
            return False

        # Lookup any HEBIs.
        if not self.hebilookup():
            self.warn("Unable to connect to any HEBIs!!!")
            return False

        # Connect to the requested HEBIs.
        if not self.hebiconnect():
            self.warn("Unable to connect to the requested HEBIs!!!")
            return False

        # And finally set the feedback rate and command lifetime.
        self.hebirate(self.rate)
        self.hebilifetime(self.lifetime)
        return True

    # Locate the HEBI actuators
    def hebilookup(self):
        # Locate HEBI actuators on the network, waiting 1s for discovery.
        self.info("Locating HEBIs...")
        self.lookup = hebi.Lookup()
        sleep(1)

        # Check we have something.
        if sum(1 for _ in self.lookup.entrylist) == 0:
            self.warn("No HEBIs located.")
            return False

        # Report the connected HEBIs.
        for entry in self.lookup.entrylist:
            self.info("Located family '%s' name '%s' at address %s" %
                      (entry.family, entry.name, entry.mac_address))
        return True

    # Connect to the HEBI group to send commands/receive feedback.
    def hebiconnect(self):
        # Use the HEBI lookup to create the actuator group.
        self.group = self.lookup.get_group_from_names([self.family],self.motors)

        # Make sure all is good.
        if self.group is None:
            self.error("Unable to connect to selected motors!")
            self.error("Make sure motors are powered/connected.")
            self.error("(for example using the Scope application)")
            return False

        # Report
        self.info("Connected to HEBIs.")
        return True

    # Set the HEBI command lifetime (in milliseconds).
    def hebilifetime(self, lifetime):
        if lifetime > 0.0:
            self.group.command_lifetime = lifetime
        else:
            self.warn("Ignoring illegal negative command lifetime!")

    # Set the HEBI feedback frequency (in Hertz).
    def hebirate(self, rate):
        if rate > 0.0:
            self.group.feedback_frequency = rate
        else:
            self.warn("Ignoring illegal negative feedback frequency!")


    ######################################################################
    # Watchdog callback
    def watchdogCB(self):
        self.warn("Not getting HEBI feedback - check connection")

    # Feedback callback - send feedback on the ROS message.
    def sendfbk(self):
        # Reset the current time.  The pos/vel/eff data was already set.
        self.fbkmsg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.fbkmsg)

        # Reset the watchdog.
        self.watchdog.reset()

    def fbkCB_real(self, fbk):
        # Transfer the pos/vel/eff info.
        self.fbkmsg.position = fbk.position.tolist()
        self.fbkmsg.velocity = fbk.velocity.tolist()
        self.fbkmsg.effort   = fbk.effort.tolist()

        # Send the HEBI feedback info.
        self.sendfbk()

    def fbkCB_sim(self):
        # The current pos/vel/eff was set by the incoming command!

        # Send the HEBI feedback info.
        self.sendfbk()

    # Command callback - process the commands.
    def cmdmsgcheck(self, cmdmsg):
        # Check the message names matching the joint names.
        if not (cmdmsg.name == self.joints):
            self.warn("Joint commands not matching joint names!")
            return False

        # Check the command dimensions.
        l  = self.N
        lp = len(cmdmsg.position)
        lv = len(cmdmsg.velocity)
        le = len(cmdmsg.effort)
        if not (lp==0 or lp==l) or not (lv==0 or lv==l) or not (le==0 or le==l):
            self.warn("Illegal length of pos/vel/eff commands!")
            return False

        # Else is ok.
        return True
    
    def cmdCB_real(self, cmdmsg):
        # Check the command message.
        if not self.cmdmsgcheck(cmdmsg):
            return

        # Copy the commands.
        lp = len(cmdmsg.position)
        lv = len(cmdmsg.velocity)
        le = len(cmdmsg.effort)
        self.cmd.position = self.nans if (lp==0) else np.array(cmdmsg.position)
        self.cmd.velocity = self.nans if (lv==0) else np.array(cmdmsg.velocity)
        self.cmd.effort   = self.nans if (le==0) else np.array(cmdmsg.effort)

        # And send.
        self.group.send_command(self.cmd)

    def cmdCB_sim(self, cmdmsg):
        # Check the command message.
        if not self.cmdmsgcheck(cmdmsg):
            return

        # Immediately save into the feedback message.
        self.fbkmsg.position = cmdmsg.position
        self.fbkmsg.velocity = cmdmsg.velocity
        self.fbkmsg.effort   = cmdmsg.effort


#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the HEBI node.
    node = HebiNode('hebinode')

    # Run the startup and spin the node until interrupted.
    try:
        node.startup()
        rclpy.spin(node)
    except BaseException as ex:
        print("Ending due to exception: %s" % repr(ex))
        print_exc()

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
