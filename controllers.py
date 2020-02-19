import numpy as np
from mathtools import Quaternion, Vector3, Matrix3

MOTOR_MAX_SPEED = 300


class AttitudeController(object):

    def __init__(self, inertia, m_inertia):
        self.inertia = inertia
        self.m_inertia = m_inertia
        self.transform = Matrix3([[1.0, 0.0, 0.0],
                                  [0.0, -1.0, 0.0],
                                  [0.0, 0.0, -1.0]])    # Transformation between RW speeds vector and speeds in the body frame
        self.inv_transform = self.transform.inv()       # To save computing time
        self.Kp = 2.0 * (4.744 / 3.0)**2
        self.Kd = 2.0 * (4.744 / 3.0)

    def compute_output(self, desired_attitude, attitude, bdy_ang_vel, motors_speed):
        # Compute the linearizing command
        nonlinear_output = - self.inv_transform * bdy_ang_vel.cross(self.inertia * bdy_ang_vel + self.m_inertia * self.transform * motors_speed) / self.m_inertia

        # Compute the linear command
        q_diff = attitude.conj() * desired_attitude
        attitude_err = Vector3(q_diff.x, q_diff.y, q_diff.z)
        linear_output = - self.inv_transform * self.inertia * (self.Kp * attitude_err - self.Kd * bdy_ang_vel) / self.m_inertia
        
        # Add both
        return linear_output + nonlinear_output



class MotorAccelController(object):

    def __init__(self, speedconst, refresh_rate):
        self.speedconst = speedconst
        self.refresh_rate = refresh_rate    # [Hz]   correspond to the refresh rate of the motor controller
        self.desired_speed = Vector3(0.0, 0.0, 0.0)

    def compute_output(self, desired_accel):
        self.desired_speed = saturation(self.desired_speed + desired_accel / self.refresh_rate,
                                        -MOTOR_MAX_SPEED,
                                        MOTOR_MAX_SPEED)
        return self.desired_speed / self.speedconst
    

def saturation(vect, min_scalar, max_scalar):
    if vect.x < min_scalar:
        vect.x = min_scalar
    elif vect.x > max_scalar:
        vect.x = max_scalar
    if vect.y < min_scalar:
        vect.y = min_scalar
    elif vect.y > max_scalar:
        vect.y = max_scalar
    if vect.z < min_scalar:
        vect.z = min_scalar
    elif vect.z > max_scalar:
        vect.z = max_scalar

    return vect
    
