import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory_3d(p_true, p_est, p_ref):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(p_true[:, 0], p_true[:, 1], p_true[:, 2], label='True')
    ax.plot(p_est[:, 0], p_est[:, 1], p_est[:, 2], label='Estimated')
    ax.plot(p_ref[:, 0], p_ref[:, 1], p_ref[:, 2], '--', label='Reference')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('3D Trajectory')
    ax.legend()

    return fig


def plot_position_tracking(t, p_true, p_est, p_ref):
    fig, axs = plt.subplots(3, 1, sharex=True)

    labels = ['x', 'y', 'z']

    for i in range(3):
        axs[i].plot(t, p_true[:, i], label=f'True {labels[i]}')
        axs[i].plot(t, p_est[:, i], label=f'Est {labels[i]}')
        axs[i].plot(t, p_ref[:, i], '--', label=f'Ref {labels[i]}')
        axs[i].set_ylabel(f'{labels[i]} [m]')
        axs[i].legend()

    axs[2].set_xlabel('Time [s]')
    fig.suptitle('Position Tracking')
    fig.tight_layout()

    return fig


def plot_position_error(t, p_true, p_est):
    pos_err = p_est - p_true
    pos_err_norm = np.linalg.norm(pos_err, axis=1)

    fig, axs = plt.subplots(4, 1, sharex=True)

    axs[0].plot(t, pos_err[:, 0])
    axs[0].set_ylabel('ex [m]')

    axs[1].plot(t, pos_err[:, 1])
    axs[1].set_ylabel('ey [m]')

    axs[2].plot(t, pos_err[:, 2])
    axs[2].set_ylabel('ez [m]')

    axs[3].plot(t, pos_err_norm)
    axs[3].set_ylabel('||e|| [m]')
    axs[3].set_xlabel('Time [s]')

    fig.suptitle('Position Estimation Error')
    fig.tight_layout()

    return fig


def plot_control_inputs(t, u):
    fig, axs = plt.subplots(4, 1, sharex=True)

    axs[0].plot(t, u[:, 0])
    axs[0].set_ylabel('Thrust [N]')

    axs[1].plot(t, u[:, 1])
    axs[1].set_ylabel('Tau_x [Nm]')

    axs[2].plot(t, u[:, 2])
    axs[2].set_ylabel('Tau_y [Nm]')

    axs[3].plot(t, u[:, 3])
    axs[3].set_ylabel('Tau_z [Nm]')
    axs[3].set_xlabel('Time [s]')

    fig.suptitle('Control Inputs')
    fig.tight_layout()

    return fig


def show_all():
    plt.show()