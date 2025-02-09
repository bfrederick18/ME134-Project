import os
import xacro

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import Shutdown
from launch_ros.actions                import Node


def generate_launch_description():
    rvizcfg = os.path.join(pkgdir('project'), 'rviz/viewurdf.rviz')

    urdf = os.path.join(pkgdir('project'), 'urdf/threedof.urdf')
    with open(urdf, 'r') as file:
        robot_description = file.read()


    node_rviz = Node(
        name       = 'rviz', 
        package    = 'rviz2',
        executable = 'rviz2',
        output     = 'screen',
        arguments  = ['-d', rvizcfg],
        on_exit    = Shutdown())

    node_robot_state_publisher = Node(
        name       = 'robot_state_publisher', 
        package    = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output     = 'screen',
        parameters = [{'robot_description': robot_description}])

    node_hebi = Node(
        name       = 'hebi', 
        package    = 'hebiros',
        executable = 'hebinode',
        output     = 'screen',
        parameters = [{'family':   'robotlab'},
                      {'motors':   ['2.5',  '2.4',      '2.3']},
                      {'joints':   ['base', 'shoulder', 'elbow']}],
        on_exit    = Shutdown())

    node_trajectory = Node(
        name       = 'trajectory', 
        package    = 'project',
        executable = 'trajectory',
        output     = 'screen')


    return LaunchDescription([
        node_rviz,
        node_robot_state_publisher,
        node_hebi,
        node_trajectory,
    ])