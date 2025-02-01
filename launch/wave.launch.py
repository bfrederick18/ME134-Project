from launch                            import LaunchDescription
from launch_ros.actions                import Node


def generate_launch_description():
    node_hebi = Node(
        name       = 'hebi', 
        package    = 'hebiros',
        executable = 'hebinode',
        output     = 'screen',
        parameters = [{'family': 'robotlab'},
                      {'motors': ['2.5', '2.4', '2.3']},
                      {'joints': ['one', 'two', 'three']}])

    node_demo = Node(
        name       = 'wave', 
        package    = 'project',
        executable = 'wave',
        output     = 'screen')


    return LaunchDescription([
        node_hebi,
        node_demo,
    ])