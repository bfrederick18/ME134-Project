'''TransformHelpers.py

   This fixes a typo in the HW3 version...


   These are helper functions for rotation and transform matrices.

   They simply convert the information between different formats.  For
   example python lists or NumPy arrays, encoding Euler Angles,
   Quaternions, or Rotation Matrices.  The formats include:

      NumPy array 3     "p"     Point vector
      NumPy array 3     "n"     Axis (unit) vector
      NumPy array 3     "e"     Error vector
      NumPy array 3x3   "R"     Rotation matrix
      NumPy array 4     "quat"  Quaternion (x,y,z,w)
      NumPy array 4x4   "T"     Transform matrix

   as well as the python list URDF tags <origin> and <axis>:

      Python List 3:    <axis>          Axis vector
      Python List 6:    <origin>        6D Transform Information
      Python List 3:    <origin> "xyz"  Vector of Positions
      Python List 3:    <origin> "rpy"  Vector of Euler Angles

   as well as the ROS message elements (types):

      ROS Message  Point        Point (x,y,z) in space
      ROS Message  Vector3      3D (x,y,z) vector
      ROS Message  Quaternion   Quaternion
      ROS Message  Pose         Point + Quaternion -> 3D Rigid Body definition
      ROS Message  Transform    Vector3 + Quaterion -> Frame to Frame shift


   The Helper functions are:

   Cross Product:   cross(a,b)          Cross product of two 3-vectors
                    crossmat(a)         Cross product matrix

   Position         pzero()             Zero position vector
                    pxyz(x,y,z)         Position vector

   Axis Vectors     nx()                Unit x-axis
                    ny()                Unit y-axis
                    nz()                Unit z-axis
                    nxyz(x,y,z)         Unit vector

   Rotation Matrix  Rotx(alpha)         Rotation matrix about x-axis
                    Roty(alpha)         Rotation matrix about y-axis
                    Rotz(alpha)         Rotation matrix about z-axis
                    Rotn(n, alpha)      Rotation matrix about unit vector n

                    Reye()              Identity rotation matrix
                    Rmid(R0,R1)         Return rotation halfway  between R0,R1

   Interpolation    pinter(p0,p1,s)     Positon  factor s between p0,p1
                    Rinter(R0,R1,s)     Rotation factor s between R0,R1 
                    vinter(p0,p1,sdot)  Linear  velocity for pinter(p0,p1,s)
                    winter(R0,R1,sdot)  Angular velocity for Rinter(R0,R1,s)

   Error Vectors    ep(pd, p)           Translational error vector
                    eR(Rd, R)           Rotational error vector

   Transforms       T_from_Rp(R,p)      Compose T matrix
                    p_from_T(T)         Extract position vector from T
                    R_from_T(T)         Extract rotation matrix from T

   Quaternions      quat_from_xyzw(x,y,z,w)        Create quat from (x,y,z,w)
                    quat_from_R(R)                 Convert R to quaternion
                    R_from_quat(quat)              Convert quaternion to R

   Axis/Angle       axisangle_from_R(R)            Convert R to (axis,angle)

   Roll/Pitch/Yaw   R_from_RPY(roll, pitch, yaw)   Construct R

   URDF Elements    T_from_URDF_origin(origin)     Construct transform
                    n_from_URDF_axis(axis)         Construct axis vector

   From ROS Msgs    p_from_Point(point)            Create p from Point msg
                    p_from_Vector3(vector3)        Create p from Vector3 msg
                    R_from_Quaternion(quaternion)  Create R from Quaternion msg
                    T_from_Pose(pose)              Create T from Pose msg
                    T_from_Transform(transform)    Create T from Transform msg

   To ROS Msgs      Point_from_p(p)                Create Point msg from p
                    Vector3_from_p(p)              Create Vector3 msg from p
                    Quaternion_from_R(R)           Create Quaternion msg from R
                    Pose_from_T(T)                 Create Pose msg from T
                    Transform_from_T(T)            Create Transform msg from T
'''

import numpy as np

from urdf_parser_py.urdf        import Robot
from geometry_msgs.msg          import Point, Vector3
from geometry_msgs.msg          import Quaternion
from geometry_msgs.msg          import Pose, Transform


#
#   Cross Product
#
def cross(a,b):
    return crossmat(a) @ b

def crossmat(a):
    return np.array([[  0.0, -a[2],  a[1]],
                     [ a[2],   0.0, -a[0]],
                     [-a[1],  a[0],  0.0]])


#
#   Position 3-Vector
#
def pzero():
    return np.zeros(3)

def pxyz(x,y,z):
    return np.array([x,y,z])


#
#   Axis (Unit 3-Vector)
#
def nx():
    return nxyz(1.0, 0.0, 0.0)
def ny():
    return nxyz(0.0, 1.0, 0.0)
def nz():
    return nxyz(0.0, 0.0, 1.0)
    
def nxyz(x,y,z):
    return np.array([x,y,z]) / np.sqrt(x*x+y*y+z*z)


#
#   3x3 Rotation Matrix
#
def Reye():
    return np.eye(3)

def Rotx(alpha):
    return np.array([[1.0, 0.0          ,  0.0          ],
                     [0.0, np.cos(alpha), -np.sin(alpha)],
                     [0.0, np.sin(alpha),  np.cos(alpha)]])

def Roty(alpha):
    return np.array([[ np.cos(alpha), 0.0, np.sin(alpha)],
                     [ 0.0          , 1.0, 0.0          ],
                     [-np.sin(alpha), 0.0, np.cos(alpha)]])

def Rotz(alpha):
    return np.array([[np.cos(alpha), -np.sin(alpha), 0.0],
                     [np.sin(alpha),  np.cos(alpha), 0.0],
                     [0.0          ,  0.0          , 1.0]])

def Rotn(n, alpha):
    nx = crossmat(n)
    return np.eye(3) + np.sin(alpha) * nx + (1.0-np.cos(alpha)) * nx @ nx


def Rmid(R0, R1):
    return Rinter(R0, R1, 0.5)


#
#   Linear Interpolation (both linear and angular)
#
def pinter(p0, p1, s):
    return p0 + (p1-p0)*s

def vinter(p0, p1, sdot):
    return      (p1-p0)*sdot

def Rinter(R0, R1, s):
    (axis, angle) = axisangle_from_R(R0.T @ R1)
    return R0 @ Rotn(axis, s*angle)

def winter(R0, R1, sdot): 
    (axis, angle) = axisangle_from_R(R0.T @ R1)
    return R0 @ axis * angle * sdot


#
#   Error 3-Vectors
#
def ep(pd, p):
    return (pd - p)

def eR(Rd, R):
    return 0.5 * (cross(R[0:3,0], Rd[0:3,0]) +
                  cross(R[0:3,1], Rd[0:3,1]) +
                  cross(R[0:3,2], Rd[0:3,2]))


#
#   4x4 Transform Matrix
#
#   Build the T matrix from R/P, or extract the R/p pieces.
#
def T_from_Rp(R, p):
    return np.vstack((np.hstack((R,p.reshape((3,1)))),
                      np.array([0.0, 0.0, 0.0, 1.0])))

def p_from_T(T):
    return T[0:3,3]
def R_from_T(T):
    return T[0:3,0:3]


#
#   Quaternions
#
#   Note quaterions are stored as [x,y,z,w]
#
#   Convert to/from a rotation matrix.
#
def quat_from_xyzw(x,y,z,w):
    return np.array([x,y,z,w])

def quat_from_R(R):
    A = [1.0 + R[0][0] + R[1][1] + R[2][2],
         1.0 + R[0][0] - R[1][1] - R[2][2],
         1.0 - R[0][0] + R[1][1] - R[2][2],
         1.0 - R[0][0] - R[1][1] + R[2][2]]
    i = A.index(max(A))
    A = A[i]
    c = 0.5/np.sqrt(A)
    if   (i == 0):
        q = c*np.array([R[2][1]-R[1][2], R[0][2]-R[2][0], R[1][0]-R[0][1], A])
    elif (i == 1):
        q = c*np.array([A, R[1][0]+R[0][1], R[0][2]+R[2][0], R[2][1]-R[1][2]])
    elif (i == 2):
        q = c*np.array([R[1][0]+R[0][1], A, R[2][1]+R[1][2], R[0][2]-R[2][0]])
    else:
        q = c*np.array([R[0][2]+R[2][0], R[2][1]+R[1][2], A, R[1][0]-R[0][1]])
    return q

def R_from_quat(quat):
    norm2 = np.inner(quat,quat)
    v     = quat[0:3]
    w     = quat[3]
    R     = (2/norm2) * (np.outer(v,v) + w*w*Reye() + w*crossmat(v)) - Reye()
    return R



#
#   Axis/Angle
#
#   Pull the axis/angle from the rotation matrix.  For numberical
#   stability, go through the quaternion representation.
#
def axisangle_from_R(R):
    quat  = quat_from_R(R)
    v     = quat[0:3]
    w     = quat[3]
    n     = np.sqrt(np.inner(v,v))
    angle = 2.0 * np.arctan2(n, w)
    if n==0: axis = np.zeros(3)
    else:    axis = np.array(v) / n
    return (axis, angle)

#
#   Roll/Pitch/Yaw
#
#   Build a rotation matrix from roll/pitch/yaw angles.  These are
#   extrinsic Tait-Bryson rotations ordered roll(x)-pitch(y)-yaw(z).
#   To compute via the rotation matrices, we use the equivalent
#   intrinsic rotations in the reverse order.
#
def R_from_RPY(roll, pitch, yaw):
    return Rotz(yaw) @ Roty(pitch) @ Rotx(roll)


#
#   URDF <origin> element
#
#   The <origin> element should contain "xyz" and "rpy" attributes:
#     origin.xyz  x/y/z coordinates of the position
#     origin.rpy  Angles for roll/pitch/yaw or x/y/z extrinsic rotations
#
def p_from_URDF_xyz(xyz):
    return np.array(xyz)

def R_from_URDF_rpy(rpy):
    return R_from_RPY(rpy[0], rpy[1], rpy[2])

def T_from_URDF_origin(origin):
    return T_from_Rp(R_from_URDF_rpy(origin.rpy), p_from_URDF_xyz(origin.xyz))


#
#   URDF <axis> element
#
#   The <axis> element should contain only an "xyz" attribute.  Being
#   a single attribute, the parser directly presents this:
#     axis        x/y/z coordinates of the axis
#
def n_from_URDF_axis(axis):
    return np.array(axis)



#
#   From ROS Messages
#
#   Extract the info (re-organizing) from ROS message types
#
def p_from_Point(point):
    return pxyz(point.x, point.y, point.z)

def p_from_Vector3(vector3):
    return pxyz(vector3.x, vector3.y, vector3.z)

def quat_from_Quaternion(quaternion):
    return np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w])

def R_from_Quaternion(quaternion):
    return R_from_quat(quat_from_Quaternion(quaternion))

def T_from_Pose(pose):
    return T_from_Rp(R_from_Quaternion(pose.orientation),
                     p_from_Point(pose.position))

def T_from_Transform(transform):
    return T_from_Rp(R_from_Quaternion(transform.rotation),
                     p_from_Vector3(transform.translation))


#
#   To ROS Messages
#
#   Construct the ROS message types (re-organizing as appropriate).
#
def Point_from_p(p):
    return Point(x=p[0], y=p[1], z=p[2])

def Vector3_from_p(p):
    return Vector3(x=p[0], y=p[1], z=p[2])

def Quaternion_from_quat(quat):
    return Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

def Quaternion_from_R(R):
    return Quaternion_from_quat(quat_from_R(R))

def Pose_from_T(T):
    return Pose(position    = Point_from_p(p_from_T(T)),
                orientation = Quaternion_from_R(R_from_T(T)))

def Transform_from_T(T):
    return Transform(translation = Vector3_from_p(p_from_T(T)),
                     rotation    = Quaternion_from_R(R_from_T(T)))


#
#   Main Test Code
#
#   Simply test the above functions to make sure we have them correct.
#
if __name__ == "__main__":
    # Prepare the print format.
    np.set_printoptions(precision=6, suppress=True)

    # Test...
    print("Rotx(45):\n",              Rotx(np.radians(45)))
    print("Axis/angle (Roty(60)):\n", axisangle_from_R(Roty(np.radians(60))))
    print("Quaterion (Rotz(90)):\n",  quat_from_R(Rotz(np.radians(90))))

    quat = quat_from_R(Rotz(np.radians(90)))
    print("R_from_quat():\n",         R_from_quat(quat))
