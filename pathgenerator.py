import numpy as np
import matplotlib.pyplot as plt


def compute_accel(desired_accel, desired_speed, duration, N):
    accel = np.zeros(int(N))
    N1 = int(desired_speed / desired_accel * N / duration)
    N2 = int((duration - desired_speed / desired_accel) * N / duration)
    if N1 <= N2:
        accel[:N1] = desired_accel
        accel[N2:] = -desired_accel
    else:
        N0 = int(2.0 * N / duration)
        accel[:N0] = desired_accel
        accel[N0:] = -desired_accel

    return accel

def compute_vel(desired_accel, desired_speed, duration, N):
    pass


if __name__=='__main__':
    duration = 3.0
    DT = 1/100.0
    N = duration / DT
    desired_accel = 1.0
    desired_speed = 1.0

    time = np.linspace(0.0, duration, N)
    accel = compute_accel(desired_accel, desired_speed, duration, N)

    plt.plot(time, accel)
    plt.show()