"""Show/Test the URDF of the robot (no actual hardware required)

   This simply drives the URDF from the GUI, bypassing any hardware.

   This should start
     1) RVIZ, ready to view the robot
     2) The robot_state_publisher (listening to /joint_states)
     3) The GUI to drive the URDF (sending to /joint_states)

"""

import os
import xacro

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import Shutdown
from launch_ros.actions                import Node


#
# Generate the Launch Description
#
def generate_launch_description():

    ######################################################################
    # LOCATE FILES

    # Locate the RVIZ configuration file.
    rvizcfg = os.path.join(pkgdir('project'), 'rviz/viewurdf.rviz')

    # Locate/load the robot's URDF file (XML).
    # urdf = os.path.join(pkgdir('threedof'), 'urdf/threedof.urdf')
    urdf = os.path.join(pkgdir('project'), 'urdf/threedof.urdf')
    with open(urdf, 'r') as file:
        robot_description = file.read()


    ######################################################################
    # PREPARE THE LAUNCH ELEMENTS

    # Configure a node for RVIZ.
    node_rviz = Node(
        name       = 'rviz', 
        package    = 'rviz2',
        executable = 'rviz2',
        output     = 'screen',
        arguments  = ['-d', rvizcfg],
        on_exit    = Shutdown())

    # Configure a node for the robot_state_publisher.
    node_robot_state_publisher = Node(
        name       = 'robot_state_publisher', 
        package    = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output     = 'screen',
        parameters = [{'robot_description': robot_description}])

    # Configure a node for the GUI to substitute for the robot.
    node_gui = Node(
        name       = 'gui', 
        package    = 'joint_state_publisher_gui',
        executable = 'joint_state_publisher_gui',
        output     = 'screen',
        on_exit    = Shutdown())


    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST
    
    # Return the description, built as a python list.
    return LaunchDescription([

        # Use RVIZ to view the URDF commanded by the GUI.
        node_rviz,
        node_robot_state_publisher,
        node_gui,
    ])
