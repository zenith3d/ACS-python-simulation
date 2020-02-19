import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mathtools import Vector3, Quaternion

# FIGURE FONT
FONT = {'family': 'serif',
        'weight': 'normal',
        'size': 12}
FONT_LEGEND = {'family': 'serif',
               'weight': 'normal',
               'size': 10}

# NED FRAME
X_BDY_FRAME = Vector3(1.01, 0.0, 0.0)
Y_BDY_FRAME = Vector3(0.0, 1.01, 0.0)
Z_BDY_FRAME = Vector3(0.0, 0.0, 1.01)


class SpherePlot(object):

    def __init__(self, name=''):
        fig = plt.figure(name, figsize=(6,5))
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        self.ax = fig.gca(projection='3d')
        self.ax.set_axis_off()
        self.ax.view_init(elev=20, azim=25)
        self.drawUnitSphere(alpha=0.1)

    def drawUnitSphere(self, alpha=0.2):
        # /!\ Can be initialized outside as general variables to save time /!\ 
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 40)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(len(u)), np.cos(v))

        # Plot the sphere surface
        self.ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='c', linewidth=0, alpha=alpha, zorder=3)

    def add_attitude(self, q, c='black', linestyle='-', label=''):
        # Compute the body frame vectors corresponding to the attitude
        X = X_BDY_FRAME.rotate(q.conj())
        Y = Y_BDY_FRAME.rotate(q.conj())
        Z = Z_BDY_FRAME.rotate(q.conj())

        # Plot X, Y and Z vectors
        self.ax.quiver(0.0, 0.0, 0.0, X.x * 1.5, X.y * 1.5, X.z * 1.5, arrow_length_ratio=0.1, color=c, linestyle=linestyle, zorder=2)
        self.ax.quiver(0.0, 0.0, 0.0, Y.x * 1.5, Y.y * 1.5, Y.z * 1.5, arrow_length_ratio=0.1, color=c, linestyle=linestyle, zorder=2)
        self.ax.quiver(0.0, 0.0, 0.0, Z.x * 1.5, Z.y * 1.5, Z.z * 1.5, arrow_length_ratio=0.1, color=c, linestyle=linestyle, zorder=2)
        self.ax.text(X.x * 1.6, X.y * 1.6, X.z * 1.6, "$x_{}$".format(label), horizontalalignment='center', verticalalignment='center', zorder=2)
        self.ax.text(Y.x * 1.6, Y.y * 1.6, Y.z * 1.6, "$y_{}$".format(label), horizontalalignment='center', verticalalignment='center', zorder=2)
        self.ax.text(Z.x * 1.6, Z.y * 1.6, Z.z * 1.6, "$z_{}$".format(label), horizontalalignment='center', verticalalignment='center', zorder=2)

    def add_path(self, path, linestyle=':', label=''):
        # Compute the X axis path
        X = [X_BDY_FRAME.rotate(q.conj()) for q in path]
        Xx = [x.x for x in X]
        Xy = [x.y for x in X]
        Xz = [x.z for x in X]
        # Compute the Y axis path
        Y = [Y_BDY_FRAME.rotate(q.conj()) for q in path]
        Yx = [y.x for y in Y]
        Yy = [y.y for y in Y]
        Yz = [y.z for y in Y]
        # Compute the Z axis path
        Z = [Z_BDY_FRAME.rotate(q.conj()) for q in path]
        Zx = [z.x for z in Z]
        Zy = [z.y for z in Z]
        Zz = [z.z for z in Z]

        # Plot the path
        self.ax.plot(Xx, Xy, Xz, color='C3', linestyle=linestyle, zorder=1, label=label)
        self.ax.plot(Yx, Yy, Yz, color='C2', linestyle=linestyle, zorder=1)
        self.ax.plot(Zx, Zy, Zz, color='C0', linestyle=linestyle, zorder=1)


def simulation_plot(time_list, d_attitude_list, attitude_list, bdy_ang_vel_list, m_speed_list, d_accel_list):
    q_w = [attitude.w for attitude in attitude_list]
    q_x = [attitude.x for attitude in attitude_list]
    q_y = [attitude.y for attitude in attitude_list]
    q_z = [attitude.z for attitude in attitude_list]

    d_q_w = [d_attitude.w for d_attitude in d_attitude_list]
    d_q_x = [d_attitude.x for d_attitude in d_attitude_list]
    d_q_y = [d_attitude.y for d_attitude in d_attitude_list]
    d_q_z = [d_attitude.z for d_attitude in d_attitude_list]

    bdy_ang_vel_x = [bdy_ang_vel.x for bdy_ang_vel in bdy_ang_vel_list]
    bdy_ang_vel_y = [bdy_ang_vel.y for bdy_ang_vel in bdy_ang_vel_list]
    bdy_ang_vel_z = [bdy_ang_vel.z for bdy_ang_vel in bdy_ang_vel_list]

    m_speed_x = [m_speed.x for m_speed in m_speed_list]
    m_speed_y = [m_speed.y for m_speed in m_speed_list]
    m_speed_z = [m_speed.z for m_speed in m_speed_list]

    d_accel_x = [v.x for v in d_accel_list]
    d_accel_y = [v.y for v in d_accel_list]
    d_accel_z = [v.z for v in d_accel_list]
    
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(10,6))
    plt.subplots_adjust(left=0.1, right=0.95, wspace=0.3, hspace=0.1)
    
    axes[0,0].plot(time_list, d_q_w, linestyle='--', linewidth=1, c='C1')
    axes[0,0].plot(time_list, d_q_x, linestyle='--', linewidth=1, c='C3')
    axes[0,0].plot(time_list, d_q_y, linestyle='--', linewidth=1, c='C2')
    axes[0,0].plot(time_list, d_q_z, linestyle='--', linewidth=1, c='C0')
    axes[0,0].plot(time_list, q_w, label="$q_w$", c='C1')
    axes[0,0].plot(time_list, q_x, label="$q_x$", c='C3')
    axes[0,0].plot(time_list, q_y, label="$q_y$", c='C2')
    axes[0,0].plot(time_list, q_z, label="$q_z$", c='C0')
    axes[0,0].set_ylabel("Attitude $\mathbf{q}$ [-]", fontdict=FONT)
    axes[0,0].grid(True, alpha=0.2)
    axes[0,0].legend(loc='upper right', prop=FONT_LEGEND)

    axes[1,0].plot(time_list, bdy_ang_vel_x, label="$X^s$", c='C3')
    axes[1,0].plot(time_list, bdy_ang_vel_y, label="$Y^s$", c='C2')
    axes[1,0].plot(time_list, bdy_ang_vel_z, label="$Z^s$", c='C0')
    axes[1,0].set_ylabel("Vit. ang. $\mathbf{\Omega}^s$ [rad/s]", fontdict=FONT)
    axes[1,0].set_xlabel("Temps [s]", fontdict=FONT)
    axes[1,0].grid(True, alpha=0.2)
    axes[1,0].legend(loc='upper right', prop=FONT_LEGEND)

    axes[0,1].plot(time_list, m_speed_x, label="RR1", c='C3')
    axes[0,1].plot(time_list, m_speed_y, label="RR2", c='C2')
    axes[0,1].plot(time_list, m_speed_z, label="RR3", c='C0')
    axes[0,1].set_ylabel("Vit. moteur $\mathbf{\omega}^s$ [rad/s]", fontdict=FONT)
    axes[0,1].grid(True, alpha=0.2)
    axes[0,1].legend(loc='upper right', prop=FONT_LEGEND)

    axes[1,1].plot(time_list, d_accel_x, label="RR1", c='C3')
    axes[1,1].plot(time_list, d_accel_y, label="RR2", c='C2')
    axes[1,1].plot(time_list, d_accel_z, label="RR3", c='C0')
    axes[1,1].set_ylabel("Commande $\dot{\mathbf{\omega}}_d^s$ [rad/s²]", fontdict=FONT)
    axes[1,1].set_xlabel("Temps [s]", fontdict=FONT)
    axes[1,1].grid(True, alpha=0.2)
    axes[1,1].legend(loc='upper right', prop=FONT_LEGEND)


if __name__ == '__main__':
    
    from mathtools import computePath

    q0 = Quaternion(1, 0, 0, 0)
    q1 = Quaternion(0.9, 0.3, 0.5, -0.2)

    slerp = computePath(q0, q1, method='SLERP', samples=20)
    lerp = computePath(q0, q1, method='LERP', samples=20)

    plot = SpherePlot('TEST')
    plot.add_attitude(q0, label="Initial")
    plot.add_attitude(q1, label="Final", linestyle='--')
    plot.add_path(slerp, c='blue', linestyle=':', label='SLERP')
    plot.add_path(lerp, c='red', linestyle=':', label='LERP')

    plt.legend()
    plt.show()
    