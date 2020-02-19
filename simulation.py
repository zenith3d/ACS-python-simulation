import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

from dynamics import Dynamics
from mathtools import Quaternion, Vector3, Matrix3, computePath
from controllers import AttitudeController, MotorAccelController
from simulationplotter import SpherePlot, simulation_plot


# ==================== Simulation parameters ====================
DT = 0.0005     # [s]   Simulation interval (RK4)
N = 40000       # []    Number of samples
print("[INFO] Simulation time : {} s ({} samples, dt={}ms)".format(DT*N, N, DT*1000))

# ===================== Path initialization =====================
INIT_ATTITUDE = Quaternion(1.0, 0.0, 0.0, 0.0).unit()

TOTAL_PATH_TIME = 10    # [s]   Total time desired to complete the path
'''
PATH = [Quaternion(0.8, 0.3, 0.2, 0.1).unit()]
'''
PATH = [Quaternion(1.0, -0.4, 0.0, 0.0).unit(),
        Quaternion(1.0, -0.4, 0.4, 0.0).unit(),
        Quaternion(1.0, 0.0, 0.4, 0.0).unit(),
        Quaternion(1.0, 0.0, 0.0, 0.0).unit()]

# =================== Dynamics initialization ===================
scarr = Dynamics()
scarr.attitude = INIT_ATTITUDE

# ================== Controllers initialization =================
MOTOR_CONTROLLER_HZ = 101.01    # [Hz]  Motorboard refresh rate (added 1.01Hz to avoid weird aliasing when simulating?)
ATT_CONTROLLER_HZ = 21.01       # [Hz]  Attitude controller refresh rate

attc = AttitudeController(scarr.inertia, scarr.m_inertia)
accc = MotorAccelController(speedconst=scarr.m_speedconst, refresh_rate=MOTOR_CONTROLLER_HZ)


# ================== Simulation initialization ==================
time_list = []
d_attitude_list = []
attitude_list = []
bdy_ang_vel_list = []
m_speed_list = []
d_accel_list = []

print("[INFO] Simulating...")
for i in range(N):

    # Refresh the time
    t = i*DT

    # Compute outputs from the main controller to be send to the motorboards
    if t % (1/ATT_CONTROLLER_HZ) < DT:
        # Follow the nice looking path
        desired_attitude = PATH[int(t/TOTAL_PATH_TIME*len(PATH)) % len(PATH)]
        
        # Compute the corresponding accelerations
        d_accels = attc.compute_output(desired_attitude,
                                       scarr.attitude, 
                                       scarr.bdy_ang_vel, 
                                       scarr.m_speed)

    # Compute the voltage outputs from the motorboards according to the desired accelerations
    if t % (1/MOTOR_CONTROLLER_HZ) < DT:
        voltages = accc.compute_output(d_accels)
    
    # Record the actual state
    time_list.append(t)
    d_attitude_list.append(desired_attitude)
    attitude_list.append(scarr.attitude)
    bdy_ang_vel_list.append(scarr.bdy_ang_vel)
    m_speed_list.append(scarr.m_speed)
    d_accel_list.append(d_accels)

    # Then, compute the next state
    scarr.compute_next_state(voltages, DT)

    # Display the progression on the console
    if i % int(N / 10) == 0:
        print("{0:.1f} %".format(i * 100 / N))

# ==================== Plotting the results =====================
print("[INFO] Plotting results...")

plot = SpherePlot('TEST')

plot.add_attitude(INIT_ATTITUDE, label='0')
for k, checkpoint in enumerate(PATH):
    plot.add_attitude(checkpoint, label=k+1, linestyle=':')

plot.add_path(attitude_list, linestyle='-', label='Simulation')
#plt.legend()
plt.grid(False)

simulation_plot(time_list, d_attitude_list, attitude_list, bdy_ang_vel_list, m_speed_list, d_accel_list)

plt.show()
