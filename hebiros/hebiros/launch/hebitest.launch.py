"""Launch the HEBI test

This instantiates the HEBI node and runs the test script.

"""

import os
import xacro

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import DeclareLaunchArgument
from launch.actions                    import IncludeLaunchDescription
from launch.actions                    import ExecuteProcess
from launch.actions                    import RegisterEventHandler
from launch.conditions                 import IfCondition
from launch.conditions                 import UnlessCondition
from launch.event_handlers             import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions              import LaunchConfiguration
from launch.substitutions              import ThisLaunchFileDir
from launch_ros.actions                import Node


#
# Generate the Launch Description
#
def generate_launch_description():

    ######################################################################
    # PREPARE THE LAUNCH ELEMENTS

    # Configure a node for the robot_state_publisher.
    node_hebi = Node(
        name       = 'hebi', 
        package    = 'hebiros',
        executable = 'hebinode',
        output     = 'screen',
        parameters = [{'simulate':  False},
                      {'family':    'robotlab'},
                      {'motors':    ['9.6', '9.1', '9.2']},
                      {'joints':    ['one', 'two', 'three']}])


    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST
    
    # Return the description, built as a python list.
    return LaunchDescription([

        # Start the hebi node.
        node_hebi,        
    ])
