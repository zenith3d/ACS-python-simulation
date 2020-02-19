import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

from dynamics import Dynamics
from mathtools import Quaternion, Vector3, computePath
from controllers import AttitudeController, MotorAccelController

# ==================== Simulation parameters ====================
DT = 0.0005
N = 30000

MOTOR_CONTROLLER_HZ = 101.01   # [Hz]  Motor controller refresh rate

print("[INFO] Simulation time : {} s ({} samples, dt={}ms)".format(DT*N, N, DT*1000))

# =================== Dynamics initialization ===================
scarr = Dynamics()

# ================== Controller initialization ==================
accc = MotorAccelController(speedconst=scarr.m_speedconst, refresh_rate=MOTOR_CONTROLLER_HZ)


# ================== Simulation initialization ==================
time_list = []
m_speed_list = []
m_accel_list = []
d_accel_list = []
voltage_list = []

print("[INFO] Simulating...")
for i in range(N):

    # Refresh the time
    t = i*DT
    
    # Motor controller
    if t % (1/MOTOR_CONTROLLER_HZ) < DT:
        d_accels = Vector3(20*np.cos(t), 0.0, 0.0)
        voltages = accc.compute_output(d_accels)
        '''
        if t < N*DT/2.0:
            d_accels = Vector3(3000.0, 0.0, 0.0)
        else:
            d_accels = Vector3(-3000.0, 0.0, 0.0)
        voltages = accc.compute_output(d_accels)
        '''

    # Record the actual state
    time_list.append(t)
    m_speed_list.append(scarr.m_speed)
    m_accel_list.append(scarr.m_current/(scarr.m_speedconst*scarr.m_inertia))
    d_accel_list.append(d_accels)
    voltage_list.append(voltages)

    # Then, compute the next state
    scarr.compute_next_state(voltages, DT)

    # Display the progression on the console
    if i % int(N / 10) == 0:
        print("{0:.1f} %".format(i * 100 / N))

# ==================== Plotting the results =====================
print("[INFO] Plotting results...")

m_speed_x = [m_speed.x for m_speed in m_speed_list]
m_accel_x = [m_accel.x for m_accel in m_accel_list]
d_accel_x = [d_accel.x for d_accel in d_accel_list]
voltage_x = [voltage.x for voltage in voltage_list]

plt.subplot(2,1,1)
plt.plot(time_list, m_speed_x)
plt.ylabel("Vitesse moteur [$rad/s$]")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(time_list, m_accel_x, label='Simulation')
plt.plot(time_list, d_accel_x, label='Commande')
plt.ylabel("Acceleration moteur [$rad/s^2$]")
plt.grid(True)
plt.legend()

'''
plt.subplot(3,1,3)
plt.plot(time_list, voltage_x, c='black')
plt.ylabel("Tension [$V$]")
plt.xlabel("Temps [$s$]")
plt.grid(True)
'''

plt.show()

import pandas as pd
data = {'time': time_list,
        'speed': m_speed_x,
        'accel': m_accel_x}
df = pd.DataFrame(data)
df.to_csv('acceleration_sim2.csv', columns=['time', 'speed', 'accel'])