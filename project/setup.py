from setuptools import find_packages, setup
from glob       import glob
from os.path    import isdir

package_name = 'project'

folders = ['launch', 'rviz', 'urdf', 'meshes']

otherfiles = []
for topfolder in folders:
    for folder in [topfolder] + \
        [f for f in glob(topfolder+'/*/', recursive=True) if isdir(f)]:
        # Grab the files in this folder and append to the mapping.
        files = [f for f in glob(folder+'/*') if not isdir(f)]
        otherfiles.append(('share/' + package_name + '/' + folder, files))


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ] + otherfiles,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'wave = project.wave:main',
            'toucharb = project.toucharb:main',
            'receivepoint = project.receivepoint:main',
            'detector = project.detector:main',
            'brain = project.brain:main',
            'trajectory = project.trajectory:main',
        ],
    },
)