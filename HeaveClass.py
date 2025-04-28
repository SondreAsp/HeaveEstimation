import numpy as np
from scipy.linalg import block_diag

class HeaveKalmanFilter:
    def __init__(self, dt, initial_omega, initial_amps, initial_phases,
            R=0.05,
            Q_base=0.12,
            offset_q=0.01):
        self.dt = dt
        self.mode_omega = initial_omega
        self.n_modes = len(initial_omega)
        self.n_states = 3 * self.n_modes + 1
        self.x = np.zeros(self.n_states)
        self.P = np.eye(self.n_states)

        # Initialize each mode's state from FFT-derived parameters.
        for i in range(self.n_modes):
            A = initial_amps[i]
            phi = initial_phases[i]
            omega = initial_omega[i]
            self.x[3*i]   = A * np.cos(phi)          # Position
            self.x[3*i+1] = -A * omega * np.sin(phi)   # Velocity
            self.x[3*i+2] = omega

        # offset state
        self.x[-1] = 1.0
        
        # Store noise parameters
        self.R = R
        self.Q_base = Q_base
        self.offset_q = offset_q
        self.update_process_noise_covariance()
    
    def update_process_noise_covariance(self):
        blocks = []
        base_block = np.eye(3) * self.Q_base  # wave state block
        for w in self.mode_omega:
            # scale the block by w, so higher freq => more process noise
            Q_i = base_block * w
            blocks.append(Q_i)
        # offset noise
        Q_offset = np.array([[self.offset_q]])
        
        self.Q = block_diag(*blocks, Q_offset)
    
    def state_transition_matrix(self):
        blocks = []
        for omega in self.mode_omega:
            c = np.cos(omega * self.dt)
            s = np.sin(omega * self.dt)
            block = np.array([[c, s/omega, 0 ],
                              [-omega * s, c, 0], 
                              [0, 0, 1]])
            blocks.append(block)
        blocks.append(np.eye(1))
        return block_diag(*blocks)
    
    def measurement_matrix(self):
        C = np.zeros(self.n_states)
        for i, omega in enumerate(self.mode_omega):
            C[3*i] = -omega**2  # Only the position parts contribute.
        C[-1] = 1.0  # The offset
        return C.reshape(1, -1)
              
    def update(self, measurement):
        
        self.update_process_noise_covariance()
        A = self.state_transition_matrix()
        self.x = A @ self.x
        self.P = A @ self.P @ A.T + self.Q
        C = self.measurement_matrix()
        pure_Heave = (C @ self.x)[0]
        y_pred = pure_Heave 
        innovation = measurement - y_pred
        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T / S
        self.x = self.x + (K.flatten() * innovation)
        # Covariance update
        I = np.eye(self.n_states)
        self.P = (I - K @ C) @ self.P @ (I - K @ C).T + (K @ np.array([[self.R]]) @ K.T)

               
    def get_heave_position(self):
        total_pos = 0.0
        for i in range(self.n_modes):
            total_pos += self.x[3*i]
        # Add the offset:
        total_pos += self.x[-1]
        return total_pos
    
    def get_heave_velocity(self):
        total_vel = 0.0
        for i in range(self.n_modes):
            total_vel += self.x[3*i+1]
        # Add the offset:
        total_vel += self.x[-1]
        return total_vel
        
    def extract_phases(self):
        phases = []
        for i in range(self.n_modes):
            vel = self.x[3*i+1]
            omega = self.x[3*i+2]
            pos = self.x[3*i]
            phase = np.arctan2(-vel / omega, pos)
            phases.append(phase)
        return np.array(phases)

    def partial_reinit(self, new_omega, new_amps, new_phases, freq_eps=0.01, phase_eps=0.2):
        old_omega = np.array(self.mode_omega)
        new_omega = np.array(new_omega)
        
        if len(old_omega) != len(new_omega):
            reinit = True
        else:
            freq_diff = np.abs(old_omega - new_omega)
            phase_diff = np.abs(np.angle(np.exp(1j * (np.array(new_phases) - np.array(self.extract_phases())))))
        
            reinit = np.any(freq_diff > freq_eps) or np.any(phase_diff > phase_eps)
        
        if reinit:
            # print("Significant mode change: reinitializing Kalman filter.")
            return HeaveKalmanFilter(self.dt, new_omega, new_amps, new_phases,
                                     R=self.R, Q_base=self.Q_base, offset_q=self.offset_q)
        else:
            # print("Minor changes in mode parameters: reusing existing filter.")
            return self
