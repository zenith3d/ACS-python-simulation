import math
import numpy as np

from mathtools import Quaternion, Matrix3, Vector3


class Dynamics(object):

    def __init__(self):

        # Global characteristics
        self.inertia = Matrix3([[0.00743, 0.00000, 0.00000],
                                [0.00000, 0.00684, 0.00000],
                                [0.00000, 0.00000, 0.00613]])     # [kg.m2]   Momentum of inertia (include fixed motors+wheels)
        self.inv_inertia = self.inertia.inv()           # To save computing time
        self.transform = Matrix3([[1.0, 0.0, 0.0],
                                  [0.0, -1.0, 0.0],
                                  [0.0, 0.0, -1.0]])    # Transformation between RW speeds vector and speeds in the body frame
        self.inv_transform = self.transform.inv()       # To save computing time

        # Motors & wheels characteristics
        self.m_resistance = 1.03        # [Ohm]         Terminal resistance
        self.m_inductance = 0.000572    # [H]           Terminal inductance
        self.m_speedconst = 27.12       # [rad.s-1.V-1] Speed constant
        self.m_inertia    = 0.0000135 +\
                            0.0001100   # [kg.m2]       Momentum of inertia (motor axle and rotor + wheel momentum of inertia)

        # Global simulation variables
        self.attitude = Quaternion(1.0,0.0,0.0,0.0) # []        Attitude quaternion relative to the absolute frame
        self.bdy_ang_vel = Vector3(0.0, 0.0, 0.0)   # [rad.s-1] Angular velocity in the body frame

        # Motors & wheels simulation variables
        self.m_speed = Vector3(0.0, 0.0, 0.0)       # [rad.s-1] Motors angular velocity
        self.m_current = Vector3(0.0, 0.0, 0.0)     # [A]       Motors current
        

    def diff(self, attitude, bdy_ang_vel, m_speed, m_current, m_voltage):
        ''' 
        Differentiation function used to compute the derivative of the state with restect to time.
        The system state is composed of : attitude (Quaternion), bdy_ang_vel, m_speed, m_current (Vector3)
        The system (only) input is m_voltage (Vector3)
        ''' 
        # Differentiate motors angular velocity and current
        diff_m_speed = m_current / (self.m_speedconst * self.m_inertia)
        diff_m_current = (m_voltage - self.m_resistance * m_current - m_speed / self.m_speedconst) / self.m_inductance

        # Differentiate the angular velocity and attitude
        diff_bdy_ang_vel = - self.inv_inertia * (self.m_inertia * self.transform * diff_m_speed +\
                                                 bdy_ang_vel.cross(self.inertia * bdy_ang_vel + self.m_inertia * self.transform * m_speed))
        diff_attitude = attitude * bdy_ang_vel / 2.0

        return diff_attitude, diff_bdy_ang_vel, diff_m_speed, diff_m_current

    def compute_next_state(self, voltages, dt):
        ''' 
        Runge-Kutta-4 implementation which use the 'diff' function
        '''
        k1_attitude, k1_bdy_ang_vel, k1_m_speed, k1_m_current = \
            self.diff(self.attitude,
                      self.bdy_ang_vel,
                      self.m_speed,
                      self.m_current,
                      voltages)
        k2_attitude, k2_bdy_ang_vel, k2_m_speed, k2_m_current = \
            self.diff(self.attitude + k1_attitude * dt / 2.0,
                      self.bdy_ang_vel + k1_bdy_ang_vel * dt / 2.0,
                      self.m_speed + k1_m_speed * dt / 2.0,
                      self.m_current + k1_m_current * dt / 2.0,
                      voltages)
        k3_attitude, k3_bdy_ang_vel, k3_m_speed, k3_m_current = \
            self.diff(self.attitude + k2_attitude * dt / 2.0,
                      self.bdy_ang_vel + k2_bdy_ang_vel * dt / 2.0,
                      self.m_speed + k2_m_speed * dt / 2.0,
                      self.m_current + k2_m_current * dt / 2.0,
                      voltages)
        k4_attitude, k4_bdy_ang_vel, k4_m_speed, k4_m_current = \
            self.diff(self.attitude + k3_attitude * dt,
                      self.bdy_ang_vel + k3_bdy_ang_vel * dt,
                      self.m_speed + k3_m_speed * dt,
                      self.m_current + k3_m_current * dt,
                      voltages)

        self.attitude = (self.attitude + dt * (k1_attitude + 2.0 * k2_attitude + 2.0 * k3_attitude + k4_attitude) / 6.0).unit()
        self.bdy_ang_vel = self.bdy_ang_vel + dt * (k1_bdy_ang_vel + 2.0 * k2_bdy_ang_vel + 2.0 * k3_bdy_ang_vel + k4_bdy_ang_vel) / 6.0
        self.m_speed = self.m_speed + dt * (k1_m_speed + 2.0 * k2_m_speed + 2.0 * k3_m_speed + k4_m_speed) / 6.0
        self.m_current = self.m_current + dt * (k1_m_current + 2.0 * k2_m_current + 2.0 * k3_m_current + k4_m_current) / 6.0
