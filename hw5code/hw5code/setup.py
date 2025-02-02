from setuptools import find_packages, setup
from glob import glob

package_name = 'hw5code'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/meshes', glob('meshes/*')),
        ('share/' + package_name + '/rviz',   glob('rviz/*')),
        ('share/' + package_name + '/urdf',   glob('urdf/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='The 133a HW5 Code',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'plotdata = hw5code.plotdata:main',
            'hw5p1    = hw5code.hw5p1:main',
            'hw5p2    = hw5code.hw5p2:main',
            'hw5p3    = hw5code.hw5p3:main',
            'hw5p4    = hw5code.hw5p4:main',
            'hw5p5    = hw5code.hw5p5:main',
        ],
    },
)
