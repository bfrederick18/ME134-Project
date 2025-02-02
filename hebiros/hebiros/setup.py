from setuptools import setup
from glob import glob

package_name = 'hebiros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/params', glob('params/*')),
        ('share/' + package_name + '/launch', glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='The 134 HEBI ROS package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'plothebi  = hebiros.plothebi:main',
            'hebinode  = hebiros.hebinode:main',
            'hebiscan  = hebiros.hebiscan:main',
            'hebitune  = hebiros.hebitune:main',
            'hebiforce = hebiros.hebiforce:main',
            'testsend  = hebiros.testsend:main',
        ],
    },
)
