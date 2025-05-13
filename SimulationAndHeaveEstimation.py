import time
import threading
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
import collections
from collections import deque
matplotlib.use('TkAgg')

# Import custom classes
from MotionMatreciesClass import MotionMatrecies
from ShipMotionClass import ShipMotion
from AttitudeClass import AttitudeClass
from HeaveClass import HeaveKalmanFilter

# ========================================================
# Update Rates and Global Constants
# ========================================================
heave_dt = 1 / 300    
attitude_dt = 1 / 300      
fs = 1.0 / attitude_dt
dt = 1 / 10

g = 9.82468  # gravitational constant at current location
d_b = np.array([3.0, 7.0, 3.5])  # Displacement vector
# Gravity in NED frame
g_vehicle = np.array([0.0, 0.0, g])

# ========================================================
# Global Variables for Timing and Data Sharing
# ========================================================
last_prediction_time = time.perf_counter()
last_correction_time = time.perf_counter()

# Global variables that store the most recent sensor and estimation data:
phi_sim_deg = 0.0
theta_sim_deg = 0.0
gyro_data = np.array([0.0, 0.0, 0.0])
acc_data = np.array([0.0, 0.0, 0.0])
P_acc_update_global = [0.0, 0.0, 0.0]
acc_zg_global = 0.0
CoG_pos_update_global = [0.0, 0.0, 0.0]
P_pos_update_global = [0.0, 0.0, 0.0]
P_posEst_update_global = 0.0
P_vel_update_global = [0.0, 0.0, 0.0]
P_velEst_update_global = 0.0
dx_global = [0.0, 0.0, 0.0]
dxy_global = [0.0, 0.0, 0.0]
dxyz_global = [0.0, 0.0, 0.0]

# Data arrays for plotting
time_data = []
phi_sim_data = []
theta_sim_data = []
phi_EKF_data = []
theta_EKF_data = []
vx_line_data = []
vy_line_data = []
vz_line_data = []
axAcc_x_data = []
axAcc_y_data = []
axAcc_z_data = []
azc_line_data = []
aze_line_data = []
axPos_z_data = []
axPosEst_z_data = []
axVel_z_data = []
axVelEst_z_data = []
heave_est_data = []
heave_vel_data = []

start_time = time.perf_counter()

# ========================================================
# EKF and IMU Setup
# ========================================================
# Initial EKF state and covariance
theta_init = phi_init = np.radians(0)
p = q = r = 0
z_ddot = 1

# Measurement noise (6x6) and process noise (6x6)
W = np.diag([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
V = np.diag([5, 5, 0.1, 0.1, 0.1, 1])

# Initial state vector and covariance matrices:
x_k = np.array([[phi_init],
                [theta_init],
                [p],
                [q],
                [r],
                [z_ddot]])
x_bar = x_k.copy()
X_bar = np.eye(6) * 1000

gyro_bias_range = 1.0 
gyro_scale_percent = 0.01
gyro_nonorth_deg   = 2.86  # assumption +/-5%
gyro_rms_noise_mdps = 75

gyro_bias = np.random.uniform(-gyro_bias_range, gyro_bias_range, size=3)
gyro_sf = 1.0 + np.random.uniform(-gyro_scale_percent, gyro_scale_percent, size=3)
Sg = np.diag(1.0 + np.random.uniform(-gyro_scale_percent, gyro_scale_percent, 3))
gyro_nonorth_rad = np.deg2rad(gyro_nonorth_deg)
Ng = np.array([
    [0, -np.random.uniform(-gyro_nonorth_rad, gyro_nonorth_rad), np.random.uniform(-gyro_nonorth_rad, gyro_nonorth_rad)],
    [np.random.uniform(-gyro_nonorth_rad, gyro_nonorth_rad), 0, -np.random.uniform(-gyro_nonorth_rad, gyro_nonorth_rad)],
    [-np.random.uniform(-gyro_nonorth_rad, gyro_nonorth_rad), np.random.uniform(-gyro_nonorth_rad, gyro_nonorth_rad), 0]
])
sigma_gyro = gyro_rms_noise_mdps / 1000

acc_bias_range = 20e-3 * g 
acc_scale_percent = 0.01
acc_nonorth_deg = 2.86  # assumption +/-5%
acc_rms_noise_mg = 2.0   

acc_bias  = np.random.uniform(-acc_bias_range, acc_bias_range, size=3)
acc_sf  = 1.0 + np.random.uniform(-acc_scale_percent, acc_scale_percent, size=3)
Sa = np.diag(1.0 + np.random.uniform(-acc_scale_percent, acc_scale_percent, 3))
acc_nonorth_rad = np.deg2rad(acc_nonorth_deg)
Na = np.array([
    [0, -np.random.uniform(-acc_nonorth_rad, acc_nonorth_rad), np.random.uniform(-acc_nonorth_rad, acc_nonorth_rad)],
    [np.random.uniform(-acc_nonorth_rad, acc_nonorth_rad), 0, -np.random.uniform(-acc_nonorth_rad, acc_nonorth_rad)],
    [-np.random.uniform(-acc_nonorth_rad, acc_nonorth_rad), np.random.uniform(-acc_nonorth_rad, acc_nonorth_rad), 0]
])
sigma_acc = acc_rms_noise_mg*g*1e-3 
sigma_acc = np.float64(sigma_acc)

# ========================================================
# Instantiate Classes
# ========================================================
motionMatrix = MotionMatrecies(g)
shipMotion = ShipMotion(roll=7, pitch=7, yaw=0, heave=2.5,
                        rollCycle=8, pitchCycle=8, yawCycle=60, heaveCycle=7, simpleWave=False)
attitude = AttitudeClass(attitude_dt, g, d_b, V, W, x_bar, X_bar)
decrease = True
acc_meas_prev = 0
gyro_meas_prev = 0

def identify_modes(signal_window, dt, min_height=0.01, max_modes=6, prominence=0.0001):

    # print(signal_window)
    signal_window = signal_window - np.mean(signal_window)
    fft_vals = rfft(signal_window)

    # print(fft_vals)
    N = len(signal_window)

    # Compute the raw amplitude and then apply one-sided scaling:
    acc_magn = np.abs(fft_vals) / N
    acc_phase = np.angle(fft_vals)
    freqs = rfftfreq(N, dt)
    omega = 2 * np.pi * freqs

    # Avoid division by zero:
    omega[0] = np.inf
    
    # Convert from acceleration amplitude to position amplitude
    pos_magn = acc_magn / (omega**2)

    # Adjust phase accordingly:
    pos_phase = acc_phase - np.pi
    
    # Peak detection on the position amplitude spectrum
    peaks_idx, res = find_peaks(pos_magn, height=min_height)

    # Sort peaks by descending amplitude:
    sorted_peaks = sorted(peaks_idx, key=lambda i: pos_magn[i], reverse=True)
    top_idx = sorted_peaks[1:max_modes]
    new_omega  = omega[top_idx]
    new_amps   = pos_magn[top_idx]
    new_phases = pos_phase[top_idx]

    return new_omega, new_amps, new_phases

def lpf(signal, prev, fc=10.0, fs=300.0):
    signal = signal.reshape((3,1))
    Ts = 1 / fs                      
    Tf = 1 / (2 * np.pi * fc)
    alpha = Ts / (Tf + Ts)
    y = (1 - alpha) * prev + alpha * signal
    return y

def Ship(t):
    global decrease, acc_meas_prev, gyro_meas_prev
    factor = 0.02
    factor2 = 0.05

    if (t > 90 and decrease): 
        shipMotion.heave -= attitude_dt*factor
        shipMotion.roll -= attitude_dt*factor2
        shipMotion.pitch -= attitude_dt*factor2
    if (t > 100 and shipMotion.heave <= 0.1 and shipMotion.heave < 2.5):
        decrease = False
    if (not decrease):
        shipMotion.heave += attitude_dt*factor
        shipMotion.roll += attitude_dt*factor2
        shipMotion.pitch += attitude_dt*factor2
        
    if shipMotion.heave <= 0:
        shipMotion.heave = 0
        shipMotion.roll = 0
        shipMotion.pitch = 0
    if shipMotion.heave >=2.5:
        shipMotion.heave = 2.5
        shipMotion.roll = 7
        shipMotion.pitch = 7
    
    if t > 110 and t < 240:
        d_b[1] += attitude_dt*0.1
    if d_b[1] >= 10:
        d_b[1] = 10
        
    if t > 250:
        d_b[1] -= attitude_dt*0.1
    if d_b[1] <= 7:
        d_b[1] = 7
    
    # Heave-based translation
    CoG_pos = [0.0, 0.0, shipMotion.z_heave(t)] 
    CoG_vel = [0.0, 0.0, shipMotion.z_heave_dot(t)]
    CoG_acc = [0.0, 0.0, shipMotion.z_heave_ddot(t)]
    
    # Angles & rates
    psi   = shipMotion.yaw_angle(t)
    theta = shipMotion.pitch_angle(t)
    phi   = shipMotion.roll_angle(t)

    psi_dot   = shipMotion.yaw_angle_dot(t)
    theta_dot = shipMotion.pitch_angle_dot(t)
    phi_dot   = shipMotion.roll_angle_dot(t)

    psi_ddot   = shipMotion.yaw_angle_ddot(t)
    theta_ddot = shipMotion.pitch_angle_ddot(t)
    phi_ddot   = shipMotion.roll_angle_ddot(t)

    # Build the DCM from the class
    C_bv = motionMatrix.c_bv(psi, theta, phi)

    # Build skew-symmetric
    Omega_mat = motionMatrix.omega(psi, theta, phi,
                                   psi_dot, theta_dot, phi_dot)
    dotOmega_mat = motionMatrix.omega_dot(psi, theta, phi,
                                          psi_ddot, theta_ddot, phi_ddot)

    # Convert offset d_b from NED->inertial
    Rd = C_bv @ d_b

    # Velocity of P
    cross_omega_Rd = Omega_mat @ Rd
    P_vel = CoG_vel + cross_omega_Rd
    
    # Define d components in inertial frame
    dx_local = np.array([d_b[0], 0, 0])
    dxy_local = np.array([d_b[0], d_b[1], 0])
    dxyz_local = d_b

    # Rotate & translate each segment
    dx = CoG_pos + C_bv @ dx_local
    dxy = CoG_pos + C_bv @ dxy_local
    dxyz = CoG_pos + C_bv @ dxyz_local

    # Angular velocity
    omega_x = Omega_mat[2, 1]
    omega_y = Omega_mat[0, 2]
    omega_z = Omega_mat[1, 0]
    omega_vec = np.array([omega_x, omega_y, omega_z])

    # Acceleration of P:
    cross_dotOmega_Rd = dotOmega_mat @ Rd
    cross_omega_omega_Rd = Omega_mat @ (Omega_mat @ Rd)
    P_acc = CoG_acc + cross_dotOmega_Rd + cross_omega_omega_Rd

    # Convert to NED frame & subtract gravity
    C_vb = C_bv.T
    aP_body = C_vb @ P_acc
    g_body = C_vb @ g_vehicle

    # Specific force (ideal accelerometer reading, no noise/bias)
    f_specific_true = aP_body - g_body

    # Position of P in vehicle frame
    P_pos = np.array(CoG_pos) + Rd

    gyro_true_deg_s = (180.0 / np.pi) * omega_vec
    gyro_noise = Sg @ (np.eye(3) + Ng) @ gyro_true_deg_s + gyro_bias + np.random.normal(0, sigma_gyro, 3)
    gyro_meas = lpf(gyro_noise, gyro_meas_prev)
    gyro_meas_prev = gyro_meas

    acc_noise = Sa @ (np.eye(3) + Na) @ f_specific_true + acc_bias + np.random.normal(0, sigma_acc, 3)
    acc_meas = lpf(acc_noise, acc_meas_prev)
    acc_meas_prev = acc_meas

    return (acc_meas, gyro_meas, phi, theta, P_pos, P_vel, P_acc, CoG_pos, dx, dxy, dxyz)
 
# --------------------------------------
# Plot Initialization
# --------------------------------------

fig = plt.figure(figsize=(22, 10))
fig.canvas.manager.set_window_title("Heave Estimator")
gs = GridSpec(3, 4, figure=fig, width_ratios=[2.5, 2, 2, 2], height_ratios=[1, 1, 1], hspace=0.6, wspace=0.6)
ax = [None, None, None, None]

# Large left-side subplot for 3D (spans vertically)
ax[0] = fig.add_subplot(gs[:, 0], projection='3d')
pos = ax[0].get_position()
ax[0].set_position([pos.x0 - 0.1, pos.y0 - 0.05, pos.width + 0.15, pos.height + 0.1])

# Top-right subplot
ax[1] = fig.add_subplot(gs[0, 1:])

# Middle-right subplot
ax[2] = fig.add_subplot(gs[1, 1:])

# Bottom-right subplot
ax[3] = fig.add_subplot(gs[2, 1:])

# 3D Ship Motion Plot
ax[0].set_title("3D Ship Motion", fontsize=14, pad=15)
ax[0].set_xlabel(r'$X$', fontsize=12, labelpad=10)
ax[0].set_ylabel(r'$Y$', fontsize=12, labelpad=10)
ax[0].set_zlabel(r'$Z$', fontsize=12, labelpad=10)
# Setting equal aspect ratio and centered axes
ax[0].set_box_aspect([8, 8, 8])
ax[0].set_xlim(-5, 5)
ax[0].set_ylim(-7, 9)
ax[0].set_zlim(-6, 8)
# Improve viewing angle
ax[0].view_init(elev=30, azim=-45)
# Grid and background
ax[0].grid(True, linewidth=0.5, alpha=0.7)
ax[0].xaxis.pane.set_edgecolor('w')
ax[0].yaxis.pane.set_edgecolor('w')
ax[0].zaxis.pane.set_edgecolor('w')
ax[0].xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.8))
ax[0].yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.8))
ax[0].zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.8))
# Plot elements 
cog_plot, = ax[0].plot([], [], [], 'ro', label=r'$CoG$', markersize=6)
p_plot,   = ax[0].plot([], [], [], 'bo', label=r'$P$', markersize=6)
line_plot,= ax[0].plot([], [], [], 'g-', label=r'$d$', linewidth=2)
# Sub-component lines
dx_line, = ax[0].plot([], [], [], 'r--', linewidth=1.5, label=r'$d_x$')
dy_line, = ax[0].plot([], [], [], 'g--', linewidth=1.5, label=r'$d_y$')
dz_line, = ax[0].plot([], [], [], 'b--', linewidth=1.5, label=r'$d_z$')
# Adjusting legend position and style
ax[0].legend(loc='upper left', fontsize=10, shadow=False)
ax[0].set_proj_type('persp')

# Heave Position (of P)
ax[1].set_title("Heave Position of P", fontsize=14, pad=15)
ax[1].set_ylim(-5, 13)
ax[1].set_xlabel(r'$Time [s]$', fontsize=12, labelpad=10)
ax[1].set_ylabel(r'$P_{z\;position} \; [m]$', fontsize=12, labelpad=10)
axPos_z, = ax[1].plot([], [], 'g-', linewidth=1, label='Calculated')
axPosEst_z, = ax[1].plot([], [], 'b-',linewidth=1, label='Estimated')
ax[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax[1].legend(loc='upper right', fontsize=10)

# Heave Velocity (of P)
ax[2].set_title("Heave Velocity of P", fontsize=14, pad=15)
ax[2].set_ylim(-7, 7)
ax[2].set_xlabel("Time [s]", fontsize=12, labelpad=10)
ax[2].set_ylabel(r'$P_{z, \; velocity} \; [m/s]$', fontsize=12, labelpad=10)
axVel_z, = ax[2].plot([], [], 'g-', linewidth=1, label='Calculated')
axVelEst_z, = ax[2].plot([], [], 'b-', linewidth=1, label='Estimated')
ax[2].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax[2].legend(loc='upper right', fontsize=10)

# Heave Acceleration (of P)
ax[3].set_title("Heave Acceleration of P", fontsize=14, pad=15)
ax[3].set_ylim(-7, 7)
ax[3].set_xlabel("Time [s]", fontsize=12, labelpad=10)
ax[3].set_ylabel(r'$P_{z, \; acceleration} \; [m/s^2]$', fontsize=12, labelpad=10)
azc_line, = ax[3].plot([], [], 'g-', linewidth=1, label='Calculated')
aze_line, = ax[3].plot([], [], 'b-', linewidth=1, label='Estimated')
ax[3].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax[3].legend(loc='upper right', fontsize=10)

# Data for animation
def update_plot(frame):
    global time_data, phi_sim_data, theta_sim_data, phi_EKF_data, theta_EKF_data
    global vx_line_data, vy_line_data, vz_line_data, axAcc_x_data, axAcc_y_data, axAcc_z_data
    global azc_line_data, aze_line_data
    global phi_sim_deg, theta_sim_deg, gyro_data, acc_data
    global P_acc_update_global, acc_zg_global, CoG_pos_update_global, P_pos_update_global
    global axPos_z_data, axPosEst_z_data, dx_global, dxy_global, dxyz_global, axVel_z_data, axVelEst_z_data

    with data_lock:
        current_time = time.monotonic() - start_time
        time_data.append(current_time)
        phi_sim_data.append(phi_sim_deg)
        theta_sim_data.append(theta_sim_deg)
        phi_EKF_data.append(x_bar[0, 0] * 180 / np.pi)
        theta_EKF_data.append(x_bar[1, 0] * 180 / np.pi)
        vx_line_data.append(gyro_data[0])
        vy_line_data.append(gyro_data[1])
        vz_line_data.append(gyro_data[2])
        axAcc_x_data.append(acc_data[0])
        axAcc_y_data.append(acc_data[1])
        axAcc_z_data.append(acc_data[2])
        azc_line_data.append(P_acc_update_global[2])
        aze_line_data.append(acc_zg_global)
        axPos_z_data.append(P_pos_update_global[2])
        axPosEst_z_data.append(P_posEst_update_global)
        axVel_z_data.append(P_vel_update_global[2])
        axVelEst_z_data.append(P_velEst_update_global)
        
    if len(time_data) > 600:
        time_data.pop(0)
        phi_sim_data.pop(0)
        theta_sim_data.pop(0)
        phi_EKF_data.pop(0)
        theta_EKF_data.pop(0)
        vx_line_data.pop(0)
        vy_line_data.pop(0)
        vz_line_data.pop(0)
        axAcc_x_data.pop(0)
        axAcc_y_data.pop(0)
        axAcc_z_data.pop(0)
        azc_line_data.pop(0)
        aze_line_data.pop(0)
        axPos_z_data.pop(0)
        axPosEst_z_data.pop(0)
        axVel_z_data.pop(0)
        axVelEst_z_data.pop(0)
   
    # Update CoG position
    cog_plot.set_data([CoG_pos_update_global[0]], [CoG_pos_update_global[1]])
    cog_plot.set_3d_properties(CoG_pos_update_global[2])
    
    # Update sensor (P) position
    p_plot.set_data([P_pos_update_global[0]], [P_pos_update_global[1]])
    p_plot.set_3d_properties(P_pos_update_global[2])
    
    # Update line from CoG to P
    line_plot.set_data([CoG_pos_update_global[0], P_pos_update_global[0]], [CoG_pos_update_global[1], P_pos_update_global[1]])
    line_plot.set_3d_properties([CoG_pos_update_global[2], P_pos_update_global[2]])
    
    # d_x: CoG to dx_global
    dx_line.set_data([CoG_pos_update_global[0], dx_global[0]], [CoG_pos_update_global[1], dx_global[1]])
    dx_line.set_3d_properties([CoG_pos_update_global[2], dx_global[2]])

    # d_y: from dx_global to dxy_global
    dy_line.set_data([dx_global[0], dxy_global[0]], [dx_global[1], dxy_global[1]])
    dy_line.set_3d_properties([dx_global[2], dxy_global[2]])

    # d_z: from dxy_global to dxyz_global
    dz_line.set_data([dxy_global[0], dxyz_global[0]], [dxy_global[1], dxyz_global[1]])
    dz_line.set_3d_properties([dxy_global[2], dxyz_global[2]])
    
    azc_line.set_data(time_data, azc_line_data)
    aze_line.set_data(time_data, aze_line_data)
    
    axVel_z.set_data(time_data, axVel_z_data)
    axVelEst_z.set_data(time_data, axVelEst_z_data)
    
    axPos_z.set_data(time_data, axPos_z_data)
    axPosEst_z.set_data(time_data, axPosEst_z_data)

    # Adjust time-axis limits for all 2D subplots
    for subplot in [ax[1], ax[2], ax[3]]:
        subplot.set_xlim(max(0, time_data[0]), time_data[-1] + 0.5)

    return (azc_line, aze_line, axPos_z, axPosEst_z, axVel_z, axVelEst_z)

ani = animation.FuncAnimation(fig, update_plot,
                              interval=int(dt*1000),
                              cache_frame_data=False)

plt.show()

# A threading lock to guard access to shared plotting data:
data_lock = threading.Lock()
heave_lock = threading.Lock()
stop_event = threading.Event()

# a "sliding buffer" for the last N samples
buffer_size = int(60 * fs)
acc_zg_window = collections.deque(maxlen=buffer_size)
buffer_full = False
heave_kf = None 

def reidentify_modes():
    global buffer_full, heave_kf
    if not buffer_full:
        return  # do nothing until the first fill
    
    arr = np.array(acc_zg_window)
    new_omega, new_amps, new_phases = identify_modes(arr, heave_dt)

    with heave_lock:
        if heave_kf is None:
            # first time  create filter
            heave_kf = HeaveKalmanFilter(heave_dt, new_omega, new_amps, new_phases)
            print(f"Heave filter created with {len(new_omega)} modes.")
        else:
            # partial reinit or in-place update
            heave_kf.partial_reinit(new_omega, new_amps, new_phases)

# -----------------------------------------------------------------------------
# Thread 1: Attitude EKF loop
# -----------------------------------------------------------------------------
def attitude_loop():
    global phi_sim_deg, theta_sim_deg, gyro_data, acc_data
    global P_acc_update_global, acc_zg_global, CoG_pos_update_global
    global P_pos_update_global, dx_global, dxy_global, dxyz_global
    global P_vel_update_global, start_time

    next_time = time.perf_counter()
    while not stop_event.is_set():
        current_time = time.perf_counter()
        if current_time >= next_time:
            # 1) Read simulation data
            t = current_time - start_time
            result = Ship(t)
            if result is not None:
                acc, gyro, phi, theta, P_pos_val, P_vel_val, P_acc_val, CoG_pos_val,\
                    dx_global, dxy_global, dxyz_global = result

                # 2) Attitude EKF
                attitude.EKF_predict()
                x = attitude.EKF_correct(acc, gyro * np.pi / 180.0)

                # 3) Calculate vertical acc in global frame
                phi_corrected, theta_corrected = x[0,0], x[1,0]
                angle_f = np.array([[phi_corrected],[theta_corrected]])
                acc_zg = motionMatrix.acc_z_global(acc[:3], angle_f)

                # 4) Store everything in global variables under lock
                with data_lock:
                    phi_sim_deg = phi * 180 / np.pi
                    theta_sim_deg = theta * 180 / np.pi
                    gyro_data = gyro
                    acc_data = acc
                    acc_zg_global = acc_zg
                    P_acc_update_global = P_acc_val
                    CoG_pos_update_global = CoG_pos_val
                    P_pos_update_global = P_pos_val
                    P_vel_update_global = P_vel_val

            # Prepare next call
            next_time += attitude_dt
        else:
            # to avoid cpu strain
            time.sleep(0.0001)

# -----------------------------------------------------------------------------
# Thread 2: Heave KF loop
# -----------------------------------------------------------------------------
def heave_loop():
    global buffer_full, acc_zg_global, P_posEst_update_global, P_velEst_update_global
    reidentify_interval = 1.0
    next_reid_time = time.perf_counter() + reidentify_interval

    next_time = time.perf_counter()
    while not stop_event.is_set():
        current_time = time.perf_counter()

        # 1) Periodic heave estimate
        if current_time >= next_time:
            # Read the latest acc_zg
            with data_lock:
                local_acc_zg = acc_zg_global

            if local_acc_zg is not None:
                if heave_kf is not None:
                    heave_kf.update(local_acc_zg)
                # Retrieve heave position and velocity
                with heave_lock:
                    if heave_kf is not None:
                        posEst_z = heave_kf.get_heave_position()
                        velEst_z = heave_kf.get_heave_velocity()
                    else:
                        posEst_z = 0.0
                        velEst_z = 0.0

                # Add to sliding buffer
                acc_zg_window.append(local_acc_zg)
                if len(acc_zg_window) == buffer_size:
                    buffer_full = True

                with data_lock:
                    P_posEst_update_global = posEst_z + d_b[2]
                    P_velEst_update_global = velEst_z

            next_time += heave_dt
        else:
            # avoid cpu strain
            time.sleep(0.0001)

        # 2) Re-identify modes at interval
        if current_time >= next_reid_time:
            reidentify_modes()
            next_reid_time += reidentify_interval

# -----------------------------------------------------------------------------
# Close event and main
# -----------------------------------------------------------------------------
def on_close(event):
    print("Figure closed, exiting.")
    stop_event.set()


def main():
    global start_time, ani
    start_time = time.monotonic()

    # Start background threads
    attitude_thread = threading.Thread(target=attitude_loop, daemon=True)
    heave_thread = threading.Thread(target=heave_loop, daemon=True)
    attitude_thread.start()
    heave_thread.start()

    # Connect the close event to on_close handler:
    fig.canvas.mpl_connect("close_event", on_close)

    # Start the animation update:
    ani = animation.FuncAnimation(fig, update_plot, interval=int(1000 / 30), cache_frame_data=False)
    plt.show()

if __name__ == '__main__':
    main()