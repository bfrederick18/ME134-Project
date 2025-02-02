from setuptools import find_packages, setup
from glob import glob

package_name = 'hw6sols'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='The 133a HW6 Solutions Code',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hw6p3asol = hw6sols.hw6p3asol:main',
            'hw6p3bsol = hw6sols.hw6p3bsol:main',
            'hw6p4sol  = hw6sols.hw6p4sol:main',
            'hw6p5sol  = hw6sols.hw6p5sol:main',
        ],
    },
)
