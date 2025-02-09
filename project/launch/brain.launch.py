import os
import xacro

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import Shutdown
from launch_ros.actions                import Node


def generate_launch_description():
    # node_usbcam = Node(
    #     name       = 'usb_cam', 
    #     package    = 'usb_cam',
    #     executable = 'usb_cam_node_exe',
    #     namespace  = 'usb_cam',
    #     output     = 'screen',
    #     parameters = [{'camera_name':  'logitech'},
    #                   {'video_device': '/dev/video0'},
    #                   {'pixel_format': 'yuyv2rgb'},
    #                   {'image_width':  640},
    #                   {'image_height': 480},
    #                   {'framerate':    15.0}])

    node_mapping = Node(
        name       = 'brain', 
        package    = 'project',
        executable = 'brain',
        output     = 'screen',
        remappings = [('/image_raw', '/usb_cam/image_raw')])

    return LaunchDescription([
        # node_usbcam,
        node_mapping,
    ])