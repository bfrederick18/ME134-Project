"""Run the touchtable code

   This launches all necessary pieces.

   This should start
     1) RVIZ, ready to view the robot
     2) The robot_state_publisher (likely listening to default /joint_states)
     3) The HEBI node to communicate with the motors
     4) The trajectory code (sending /joint_commands at 100Hz)

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
    rvizcfg = os.path.join(pkgdir('snakes_and_ladders'), 'rviz/viewurdf.rviz')

    # Locate/load the robot's URDF file (XML).
    # urdf = os.path.join(pkgdir('threedof'), 'urdf/threedof.urdf')
    urdf = os.path.join(pkgdir('snakes_and_ladders'), 'urdf/fourdof.urdf')
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

    # Configure a node for the hebi interface.
    node_hebi = Node(
        name       = 'hebi', 
        package    = 'hebiros',
        executable = 'hebinode',
        output     = 'screen',
        parameters = [{'family':   'robotlab'},
                      {'motors':   ['2.5',  '2.7', '2.4', '2.1']},
                      {'joints':   ['base', 'shoulder', 'elbow', 'wrist']}],
        on_exit    = Shutdown())

    # Configure a trajectory node.  PLACEHOLDER FOR YOUR CODE!!
    node_trajectory = Node(
        name       = 'touchtable', 
        package    = 'snakes_and_ladders',
        executable = 'touchtable',
        output     = 'screen')


    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST
    
    # Return the description, built as a python list.
    return LaunchDescription([

        # Drive the robot from the touchtable code.
        node_rviz,
        node_robot_state_publisher,
        node_hebi,
        node_trajectory,
    ])
