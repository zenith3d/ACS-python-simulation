import math
import numpy as np
from numpy import linalg

class Vector3(object):

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return '({}, {}, {})'.format(self.x, self.y, self.z)

    def __neg__(self):
        return Vector3(-self.x, -self.y, -self.z)

    def __abs__(self):
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

    def __add__(self, v):
        return Vector3(self.x + v.x, self.y + v.y, self.z + v.z)

    def __sub__(self, v):
        return Vector3(self.x - v.x, self.y - v.y, self.z - v.z)

    def __mul__(self, f):
        if isinstance(f, Quaternion):
            return Quaternion(0.0, self.x, self.y, self.z) * f
        else:
            return Vector3(self.x * f, self.y * f, self.z * f)

    def __rmul__(self, f):
        return Vector3(self.x * f, self.y * f, self.z * f)

    def __truediv__(self, f):
        return Vector3(self.x / f, self.y / f, self.z / f)

    def sqr(self):
        return self.x*self.x + self.y*self.y + self.z*self.z

    def cross(self, v):
        return Vector3(self.y * v.z - self.z * v.y,
                       self.z * v.x - self.x * v.z,
                       self.x * v.y - self.y * v.x)

    def rotate(self, q):
        q_unit = q.unit()
        new_q = q_unit.conj() * Quaternion(0.0, self.x, self.y, self.z) * q_unit
        return Vector3(new_q.x, new_q.y, new_q.z)


class Quaternion(object):

    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return '({}, {}, {}, {})'.format(self.w, self.x, self.y, self.z)

    def __neg__(self):
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def __abs__(self):
        return math.sqrt(self.sqr())

    def __add__(self, q):
        return Quaternion(self.w + q.w, 
                          self.x + q.x,
                          self.y + q.y, 
                          self.z + q.z)

    def __sub__(self, q):
        return Quaternion(self.w - q.w, 
                          self.x - q.x,
                          self.y - q.y, 
                          self.z - q.z)

    def __mul__(self, q):
        if isinstance(q, Quaternion):
            return Quaternion(self.w * q.w - self.x * q.x - self.y * q.y - self.z * q.z,
                              self.w * q.x + self.x * q.w + self.y * q.z - self.z * q.y,
                              self.w * q.y - self.x * q.z + self.y * q.w + self.z * q.x,
                              self.w * q.z + self.x * q.y - self.y * q.x + self.z * q.w)
        elif isinstance(q, Vector3):
            return Quaternion(-self.x * q.x - self.y * q.y - self.z * q.z,
                              self.w * q.x + self.y * q.z - self.z * q.y,
                              self.w * q.y - self.x * q.z + self.z * q.x,
                              self.w * q.z + self.x * q.y - self.y * q.x)
        else:
            return Quaternion(self.w * q,
                              self.x * q,
                              self.y * q,
                              self.z * q)

    def __rmul__(self, v):
        return Quaternion(self.w * v,
                          self.x * v,
                          self.y * v,
                          self.z * v)

    def __truediv__(self, q):
        if isinstance(q, Quaternion):
            return self * q.conj()
        else:
            return Quaternion(self.w / q,
                              self.x / q,
                              self.y / q,
                              self.z / q)
    
    def sqr(self):
        return self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z

    def conj(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def unit(self):
        return self / abs(self)

    def angles(self):
        # Compute roll
        roll = np.arctan2(-2*(self.y * self.z + self.w * self.x), 2 * (self.x * self.x + self.y * self.y) - 1)
        # Compute pitch (handling +/- 90° singularity)
        sin = 2 * (self.x * self.z - self.w * self.y)
        pitch = math.copysign(math.pi/2.0, sin) if abs(sin) >= 0.999 else np.arcsin(sin)
        # Compute yaw
        yaw = np.arctan2(-2*(self.x * self.y + self.z * self.w), 1 - 2*(self.y * self.y + self.z * self.z))
        return Vector3(roll, pitch, yaw)


class Matrix3(object):

    def __init__(self, array):

        if len(array)==3 and len(array[0])==3 and len(array[1])==3 and len(array[2])==3:
            self.mat = array
        else:
        	raise ValueError('Impossible to create a Matrix3 : bad shape')

    def __str__(self):
        return '┌ {:.3f}, {:.3f}, {:.3f} ┐\n'.format(self.mat[0][0], self.mat[0][1], self.mat[0][2]) + \
               '│ {:.3f}, {:.3f}, {:.3f} │\n'.format(self.mat[1][0], self.mat[1][1], self.mat[1][2]) + \
               '└ {:.3f}, {:.3f}, {:.3f} ┘'.format(self.mat[2][0], self.mat[2][1], self.mat[2][2])

    def __neg__(self):
        return Matrix3([[-self.mat[0][0], -self.mat[0][1], -self.mat[0][2]],
        				[-self.mat[1][0], -self.mat[1][1], -self.mat[1][2]],
        				[-self.mat[2][0], -self.mat[2][1], -self.mat[2][2]]])

    def __add__(self, m):
        return Matrix3([[self.mat[0][0] + m.mat[0][0], self.mat[0][1] + m.mat[0][1], self.mat[0][2]] + m.mat[0][2],
        				[self.mat[1][0] + m.mat[1][0], self.mat[1][1] + m.mat[1][1], self.mat[1][2]] + m.mat[1][2],
        				[self.mat[2][0] + m.mat[2][0], self.mat[2][1] + m.mat[2][1], self.mat[2][2]] + m.mat[2][2]])

    def __sub__(self, v):
        return Matrix3([[self.mat[0][0] - m.mat[0][0], self.mat[0][1] - m.mat[0][1], self.mat[0][2]] - m.mat[0][2],
        				[self.mat[1][0] - m.mat[1][0], self.mat[1][1] - m.mat[1][1], self.mat[1][2]] - m.mat[1][2],
        				[self.mat[2][0] - m.mat[2][0], self.mat[2][1] - m.mat[2][1], self.mat[2][2]] - m.mat[2][2]])

    def __mul__(self, m):
    	if isinstance(m, Matrix3):
    		return Matrix3([[self.mat[0][0] * m.mat[0][0] + self.mat[0][1] * m.mat[1][0] + self.mat[0][2] * m.mat[2][0],
    						 self.mat[0][0] * m.mat[0][1] + self.mat[0][1] * m.mat[1][1] + self.mat[0][2] * m.mat[2][1], 
    						 self.mat[0][0] * m.mat[0][2] + self.mat[0][1] * m.mat[1][2] + self.mat[0][2] * m.mat[2][2]],
    						[self.mat[1][0] * m.mat[1][0] + self.mat[1][1] * m.mat[1][0] + self.mat[1][2] * m.mat[2][0],
    						 self.mat[1][0] * m.mat[1][1] + self.mat[1][1] * m.mat[1][1] + self.mat[1][2] * m.mat[2][1], 
    						 self.mat[1][0] * m.mat[1][2] + self.mat[1][1] * m.mat[1][2] + self.mat[1][2] * m.mat[2][2]],
    						[self.mat[2][0] * m.mat[0][0] + self.mat[2][1] * m.mat[1][0] + self.mat[2][2] * m.mat[2][0],
    						 self.mat[2][0] * m.mat[0][1] + self.mat[2][1] * m.mat[1][1] + self.mat[2][2] * m.mat[2][1], 
    						 self.mat[2][0] * m.mat[0][2] + self.mat[2][1] * m.mat[1][2] + self.mat[2][2] * m.mat[2][2]]])

    	elif isinstance(m, Vector3):
    		return Vector3(self.mat[0][0] * m.x + self.mat[0][1] * m.y + self.mat[0][2] * m.z,
    					   self.mat[1][0] * m.x + self.mat[1][1] * m.y + self.mat[1][2] * m.z,
    					   self.mat[2][0] * m.x + self.mat[2][1] * m.y + self.mat[2][2] * m.z)

    def __rmul__(self, s):
        return Matrix3([[self.mat[0][0]*s, self.mat[0][1]*s, self.mat[0][2]*s],
        				[self.mat[1][0]*s, self.mat[1][1]*s, self.mat[1][2]*s],
        				[self.mat[2][0]*s, self.mat[2][1]*s, self.mat[2][2]*s]])

    def __truediv__(self, s):
        return Matrix3([[self.mat[0][0]/s, self.mat[0][1]/s, self.mat[0][2]/s],
        				[self.mat[1][0]/s, self.mat[1][1]/s, self.mat[1][2]/s],
        				[self.mat[2][0]/s, self.mat[2][1]/s, self.mat[2][2]/s]])

    def transp(self):
        return Matrix3([[self.mat[0][0], self.mat[1][0], self.mat[2][0]],
                        [self.mat[0][1], self.mat[1][1], self.mat[2][1]],
                        [self.mat[0][2], self.mat[1][2], self.mat[2][2]]])

    def inv(self):
        return Matrix3(linalg.inv(self.mat))


# ================ Quaternions interpolation methods ================

def slerp_method(q0, q1, t):
    # Normalize to avoid undefined behavior.
    q0_unit = q0.unit()
    q1_unit = q1.unit()

    # Compute angle between the two quaternions
    dot = q0_unit.x * q1_unit.x + q0_unit.y * q1_unit.y + q0_unit.z * q1_unit.z
    if dot < 0.0:
        q0_unit = -q1_unit
        dot = -dot
    
    theta0 = np.arccos(dot)
    theta = theta0 * t
    
    s0 = np.cos(theta) - dot * np.sin(theta) / np.sin(theta0)
    s1 = np.sin(theta) / np.sin(theta0)
    return q0_unit * s0 + q1_unit * s1

def lerp_method(q0, q1, t):
    # Normalize to avoid undefined behavior.
    q0_unit = q0.unit()
    q1_unit = q1.unit()

    return (q0_unit * (1.0 - t) + q1_unit * t).unit()

def computePath(q0, q1, method='SLERP', samples=20):
    path_t = np.linspace(0.0, 1.0, samples)
    if method == 'SLERP':
        return [slerp_method(q0, q1, t) for t in path_t]
    elif method == 'LERP' :
        return [lerp_method(q0, q1, t) for t in path_t]
