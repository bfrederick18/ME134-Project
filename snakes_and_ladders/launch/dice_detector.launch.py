"""Launch the USB camera node and dice detector.
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
    # PREPARE THE LAUNCH ELEMENTS

    # Configure the USB camera node
    node_usbcam = Node(
        name       = 'side_cam', 
        package    = 'usb_cam',
        executable = 'usb_cam_node_exe',
        namespace  = 'side_cam',
        output     = 'screen',
        parameters = [{'camera_name':  'logitech'},
                      {'video_device': '/dev/video2'},
                      {'pixel_format': 'yuyv2rgb'},
                      {'image_width':  640},
                      {'image_height': 480},
                      {'framerate':    15.0},
                      {'auto_white_balance': False},
                      {'white_balance': 3500},
                      {'auto_exposure': False},
                      {'exposure': 100},
                      {'autofocus': False}])

    # Configure the ball detector node
    node_detector = Node(
        name       = 'dice_detector', 
        package    = 'snakes_and_ladders',
        executable = 'dice_detector',
        output     = 'screen',
        remappings = [('/image_raw', '/side_cam/image_raw')])


    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST
    
    # Return the description, built as a python list.
    return LaunchDescription([

        # Start the nodes.
        node_usbcam,
        node_detector,
    ])
