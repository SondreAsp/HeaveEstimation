import numpy as np

class ShipMotion: 
    def __init__(self, roll, pitch, yaw, heave, rollCycle, pitchCycle, yawCycle, heaveCycle, simpleWave):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.heave = heave
        self.rollCycle = rollCycle
        self.pitchCycle = pitchCycle
        self.yawCycle = yawCycle
        self.heaveCycle = heaveCycle
        self.simpleWave = simpleWave
        
   # Heave motion and its derivatives
    def z_heave(self, t):
        if self.simpleWave: 
            A_heave = self.heave
            omega_heave = 2.0 * np.pi / self.heaveCycle
            return A_heave * np.sin(omega_heave * t)
        else: 
            A1 = self.heave * 1
            A2 = self.heave * 0.5
            A3 = self.heave * 0.25
            omega_1 = 2.0 * np.pi / self.heaveCycle * 1
            omega_2 = 2.0 * np.pi / self.heaveCycle * 1.5
            omega_3 = 2.0 * np.pi / self.heaveCycle * 2
            w1 = A1 * np.sin(omega_1 * t)
            w2 = A2 * np.sin(omega_2 * t)
            w3 = A3 * np.sin(omega_3 * t)
            return w1 + w2 + w3

    def z_heave_dot(self, t):
        if self.simpleWave: 
            A_heave = self.heave
            omega_heave = 2.0 * np.pi / self.heaveCycle
            return A_heave * omega_heave * np.cos(omega_heave * t)
        else:
            A1 = self.heave * 1
            A2 = self.heave * 0.5
            A3 = self.heave * 0.25
            omega_1 = 2.0 * np.pi / self.heaveCycle * 1
            omega_2 = 2.0 * np.pi / self.heaveCycle * 1.5
            omega_3 = 2.0 * np.pi / self.heaveCycle * 2
            w1 = A1 * omega_1 * np.cos(omega_1 * t)
            w2 = A2 * omega_2 * np.cos(omega_2 * t)
            w3 = A3 * omega_3 * np.cos(omega_3 * t)
            return w1 + w2 + w3

    def z_heave_ddot(self, t):
        if self.simpleWave: 
            A_heave = self.heave
            omega_heave = 2.0 * np.pi / self.heaveCycle
            return -A_heave * (omega_heave**2) * np.sin(omega_heave * t)
        else:
            A1 = self.heave * 1
            A2 = self.heave * 0.5
            A3 = self.heave * 0.25
            omega_1 = 2.0 * np.pi / self.heaveCycle * 1
            omega_2 = 2.0 * np.pi / self.heaveCycle * 1.5
            omega_3 = 2.0 * np.pi / self.heaveCycle * 2
            w1 = -A1 * (omega_1**2) * np.sin(omega_1 * t)
            w2 = -A2 * (omega_2**2) * np.sin(omega_2 * t)
            w3 = -A3 * (omega_3**2) * np.sin(omega_3 * t)
            return w1 + w2 + w3

    # Euler angles: roll, pitch, yaw
    def roll_angle(self, t):
        amp_deg = self.roll
        amp_rad = np.deg2rad(amp_deg)
        two_pi_f = 2.0 * np.pi / self.rollCycle
        return amp_rad * np.sin(two_pi_f * t)

    def pitch_angle(self, t):
        amp_deg = self.pitch
        amp_rad = np.deg2rad(amp_deg)
        two_pi_f = 2.0 * np.pi / self.pitchCycle
        return amp_rad * np.sin(two_pi_f * t)

    def yaw_angle(self, t):
        amp_deg = self.yaw
        amp_rad = np.deg2rad(amp_deg)
        two_pi_f = 2.0 * np.pi / self.yawCycle
        return amp_rad * np.sin(two_pi_f * t)

    def roll_angle_dot(self, t):
        amp_deg = self.roll
        amp_rad = np.deg2rad(amp_deg)
        two_pi_f = 2.0 * np.pi /self.rollCycle
        return amp_rad * two_pi_f * np.cos(two_pi_f * t)

    def pitch_angle_dot(self, t):
        amp_deg = self.pitch
        amp_rad = np.deg2rad(amp_deg)
        two_pi_f = 2.0 * np.pi / self.pitchCycle
        return amp_rad * two_pi_f * np.cos(two_pi_f * t)

    def yaw_angle_dot(self, t):
        amp_deg = self.yaw
        amp_rad = np.deg2rad(amp_deg)
        two_pi_f = 2.0 * np.pi / self.yawCycle
        return amp_rad * two_pi_f * np.cos(two_pi_f * t)

    def roll_angle_ddot(self, t):
        amp_deg = self.roll
        amp_rad = np.deg2rad(amp_deg)
        two_pi_f = 2.0 * np.pi / self.rollCycle
        return -amp_rad * (two_pi_f**2) * np.sin(two_pi_f * t)

    def pitch_angle_ddot(self, t):
        amp_deg = self.pitch
        amp_rad = np.deg2rad(amp_deg)
        two_pi_f = 2.0 * np.pi / self.pitchCycle
        return -amp_rad * (two_pi_f**2) * np.sin(two_pi_f * t)

    def yaw_angle_ddot(self, t):
        amp_deg = self.yaw
        amp_rad = np.deg2rad(amp_deg)
        two_pi_f = 2.0 * np.pi / self.yawCycle
        return -amp_rad * (two_pi_f**2) * np.sin(two_pi_f * t)
    