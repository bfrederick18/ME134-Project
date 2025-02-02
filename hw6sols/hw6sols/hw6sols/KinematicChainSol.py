'''KinematicChainSol.py

   This is the solution code for Kinematic Chains (HW6 Problem 3).

   chain = KinematicChain(node, basefame, tipframe, expectedjointnames)

      Initialize the kinematic chain, reading from the URDF message on
      the topic '/robot_description', sent by the robot_state_publisher.
      Determine the kinematic steps walking from the baseframe to the
      tipframe.  This expects the active joints to match the given names.

   (ptip, Rtip, Jv, Jw) = chain.fkin(q)

      Compute the forward kinematics and report the results.


   Node:        /kintest or as given
   Subscribe:   /robot_description      std_msgs/String

'''

import enum
import rclpy
import numpy as np

from rclpy.node                 import Node
from rclpy.qos                  import QoSProfile, DurabilityPolicy
from std_msgs.msg               import String
from urdf_parser_py.urdf        import Robot

# Grab the utilities
from hw5code.TransformHelpers   import *


#
#   Single URDF Step
#
#   This captures a single step from one frame to the next.  It be of type:
#
#     FIXED     Just a fixed T-matrix shift, nothing moving, not a DOF.
#     REVOLUTE  A fixed T-matrix shift, followed by a rotation about an axis.
#     LINEAR    A fixed T-matrix shift, followed by a transation along an axis.
#
#   It contains several pieces of permanent data (coming from the URDF):
#
#     name      String showing the name
#     type      One of the above
#     Tshift    Fixed shift: Transform of this frame w.r.t. previous
#     nlocal    Joint axis (if applicable) in the local frame
#
#   We also add information how this relates to the active joints:
#
#     dof       If an active dof (not FIXED), the dof number
#

# Define the joint types.
class JointType(enum.Enum):
    FIXED    = 0
    REVOLUTE = 1
    LINEAR   = 2

# Define a single step in the URDF (kinematic chain).
class URDFStep():
    def __init__(self, name, type, Tshift, nlocal):
        # Store the permanent/fixed/URDF data.
        self.name   = name      # Joint name
        self.type   = type      # Joint type (per above enumeration)
        self.Tshift = Tshift    # Transform w.r.t. previous frame
        self.nlocal = nlocal    # Joint axis in the local frame

        # Match against the joint numbers
        self.dof    = None      # Joint DOF number (or None if FIXED)


#
#   Kinematic Chain Object
#
#   This stores the information provided by the URDF in the form of
#   steps (see above).  In particular, see the fkin() function, as it
#   walks up the chain to determine the transforms.
#

# Define the full kinematic chain
class KinematicChain():
    # Helper functions for printing info and errors.
    def info(self, string):
        self.node.get_logger().info("KinematicChain: " + string)
    def error(self, string):
        self.node.get_logger().error("KinematicChain: " + string)
        raise Exception(string)

    # Initialization - load the URDF!
    def __init__(self, node, baseframe, tipframe, expectedjointnames):
        # Store the node (for the helper functions).
        self.node = node

        # Prepare the list the steps.
        self.steps = []
        self.dofs  = 0

        # Create a temporary subscriber to receive the URDF.  We use
        # the TRANSIENT_LOCAL durability, so that we see the last
        # message already published (if any).
        self.info("Waiting for the URDF to be published...")
        self.urdf = None
        def cb(msg):
            self.urdf = msg.data
        topic   = '/robot_description'
        quality = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             depth=1)
        sub = self.node.create_subscription(String, topic, cb, quality)
        while self.urdf is None:
            rclpy.spin_once(self.node)
        self.node.destroy_subscription(sub)

        # Convert the URDF string into a Robot object and report.
        robot = Robot.from_xml_string(self.urdf)
        self.info("Proccessing URDF for robot '%s'" % robot.name)

        # Parse the Robot object into a list of URDF steps from the
        # base frame to the tip frame.  Search backwards, as the robot
        # could be a tree structure: while a parent may have multiple
        # children, every child has only one parent.  The resulting
        # chain of steps is unique.
        frame = tipframe
        while (frame != baseframe):
            # Look for the URDF joint to the parent frame.
            joint = next((j for j in robot.joints if j.child == frame), None)
            if (joint is None):
                self.error("Unable find joint connecting to '%s'" % frame)
            if (joint.parent == frame):
                self.error("Joint '%s' connects '%s' to itself" %
                           (joint.name, frame))
            frame = joint.parent

            # Check the type (use the above enumeration)
            if  (joint.type == 'revolute' or
                 joint.type == 'continuous'):  type = JointType.REVOLUTE
            elif joint.type == 'prismatic':    type = JointType.LINEAR
            elif joint.type == 'fixed':        type = JointType.FIXED
            else:
                self.error("Joint '%s' has unknown type '%s'" %
                           (joint.name, joint.type))

            # Check that the axis is normalized.
            if type is JointType.FIXED:
                nlocal = None
            else:
                nlocal = n_from_URDF_axis(joint.axis)
                mag = np.sqrt(np.inner(nlocal, nlocal))
                if abs(mag - 1) > 1e-6:
                    self.info("WARNING Joint '%s' axis needed normalization" %
                              (joint.name))
                nlocal = nlocal / mag

            # Convert the URDF information into a single URDF step.
            # Note the axis (nlocal) is meaningless for a fixed joint.
            self.steps.insert(0, URDFStep(
                name   = joint.name,
                type   = type,
                Tshift = T_from_URDF_origin(joint.origin),
                nlocal = nlocal))

        # Set the active DOF numbers walking up the steps.
        dof = 0
        for step in self.steps:
            if step.type != JointType.FIXED:
                step.dof = dof
                dof += 1
            else:
                step.dof = None
        self.dofs = dof
        self.info("URDF has %d steps, %d active DOFs:" %
                  (len(self.steps), self.dofs))

        # Report what we found.
        for (i, step) in enumerate(self.steps):
            string = "Step #%d %-8s " % (i, step.type.name)
            string += "      " if step.dof is None else "DOF #%d" % step.dof
            string += " '%s'" % step.name
            self.info(string)

        # Confirm the active joint names matches the expectation
        jointnames = [s.name for s in self.steps if s.type != JointType.FIXED]
        if jointnames != list(expectedjointnames):
            self.error("Chain does not match the expected names: " +
                       str(expectedjointnames))


    # Compute the forward kinematics!
    def fkin(self, q):
        # Check the number of joints
        if (len(q) != self.dofs):
            self.error("Number of joint angles (%d) does not chain (%d)" %
                       (len(q), self.dofs))

        # We will build up three lists.  For each non-fixed (real)
        # joint remember the type, position, axis w.r.t. world.
        type = []
        p    = []
        n    = []

        # Initialize the T matrix to walk up the chain, w.r.t. world frame!
        T = np.eye(4)

        # Walk the chain, one URDF step at a time, adjusting T as we
        # go.  This could be a fixed or active joint.
        for step in self.steps:
            # Take action based on the joint type: Move the transform
            # T up the kinematic chain (remaining w.r.t. world frame).
            if step.type is JointType.REVOLUTE:
                # Revolute is a shift followed by a rotation:
                T = T @ step.Tshift
                T = T @ T_from_Rp(Rotn(step.nlocal, q[step.dof]), pzero())
            elif step.type is JointType.LINEAR:
                # Linear is a shift followed by a translation:
                T = T @ step.Tshift
                T = T @ T_from_Rp(Reye(), step.nlocal * q[step.dof])
            else:
                # Fixed is only a shift.
                T = T @ step.Tshift


            # For active joints, store the type, positon and axis info
            # (w.r.t. world frame).
            if step.type != JointType.FIXED:
                type.append(step.type)
                p.append(p_from_T(T))
                n.append(R_from_T(T) @ step.nlocal)

        # Collect the tip information.
        ptip = p_from_T(T)
        Rtip = R_from_T(T)

        # Collect the jacobian for each active joint.
        Jv = np.zeros((3,self.dofs))
        Jw = np.zeros((3,self.dofs))
        for i in range(self.dofs):
            # Fill in the appropriate Jacobian column based on the
            # type.  The Jacobian (like the data) is w.r.t. world.
            if type[i] is JointType.REVOLUTE:
                # Revolute is a rotation:
                Jv[:,i] = cross(n[i], ptip - p[i])
                Jw[:,i] = n[i]
            elif type[i] is JointType.LINEAR:
                # Linear is a translation:
                Jv[:,i] = n[i]
                Jw[:,i] = np.zeros(3)

        # Return the info
        return (ptip, Rtip, Jv, Jw)
