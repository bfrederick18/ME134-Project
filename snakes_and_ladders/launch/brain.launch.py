import os
import xacro

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import Shutdown
from launch_ros.actions                import Node


def generate_launch_description():

    node_mapping = Node(
        name       = 'brain', 
        package    = 'snakes_and_ladders',
        executable = 'brain',
        output     = 'screen')
        #remappings = [('/image_raw', '/usb_cam/image_raw')])

    return LaunchDescription([
        # node_usbcam,
        node_mapping,
    ])