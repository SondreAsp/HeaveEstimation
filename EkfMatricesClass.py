# -*- coding: utf-8 -*-
import numpy as np
class EkfMatrices:
    
    def __init__(self, dt, g, d):
        self.dt = dt
        self.g = g
        self.d = d
        self.prev_angular_velocity = None
           
    def g_matrix_ex_6d(self, phi, theta, p, q, r, z_ddot):
        # Ensure the angular rates are floats
        p = float(p)
        q = float(q)
        r = float(r)
          
        g = self.g
        dt = self.dt
        d = self.d
    
        # Gravity component
        gm = np.array([[-g * np.sin(theta)],
                        [ g * np.cos(theta) * np.sin(phi)],
                        [ g * np.cos(theta) * np.cos(phi)]])
        
        d = np.array(d).reshape((3, 1))
    
        # Skew-symmetric matrix for omega
        omega = np.array([[0,   -r,    q],
                          [r,    0,   -p],
                          [-q,   p,    0]])
        
        # Compute centripetal acc
        omega_cent = omega @ (omega @ d)
    
        # Compute angular acceleration (omega_dot) using finite differences.
        if self.prev_angular_velocity is None:
            # For the first measurement, assume zero angular acceleration.
            p_dot, q_dot, r_dot = 0.0, 0.0, 0.0
        else:
            prev = self.prev_angular_velocity
            p_dot = (p - prev[0]) / dt
            q_dot = (q - prev[1]) / dt
            r_dot = (r - prev[2]) / dt
            
        
        a_CoG_b = np.array([[0],
                              [0],
                              [z_ddot]])
 
        # Update stored values for the next call
        self.prev_angular_velocity = np.array([p, q, r])
    
        
        omega_dot = np.zeros((3,3))
        
        # Tangential acceleration
        omega_tan = omega_dot @ d
    
        # Total predicted acceleration at the sensor location
        ap = gm + omega_cent + omega_tan - a_CoG_b
    
        gyro_pred = np.array([[p],[q],[r]])
        return np.vstack((ap, gyro_pred)) 
        
    def deltag_ex_6d(self, phi, theta, p, q, r, z_ddot):
        p = float(p)
        q = float(q)
        r = float(r)
        z_ddot = float(z_ddot)
    
        g = self.g
        dt = self.dt
        d = self.d
        
        d1 = d[0]
        d2 = d[1]
        d3 = d[2]
        
        gyro_jacobian = np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0], 
            [0, 0, 0, 0, 1, 0]   
        ])
               
        dg_ex = np.array([
            [0, -g*np.cos(theta), d2*q + d3*r, -2*d1*q + d2*p, -2*d1*r + d3*p, 0],
            [g*np.cos(phi)*np.cos(theta), -g*np.sin(phi)*np.sin(theta), d1*q - 2*d2*p, d1*p + d3*r, -2*d2*r + d3*q, 0],
            [g*np.sin(phi)*np.cos(theta), g*np.sin(theta)*np.cos(phi), d1*r - 2*d3*p, d2*r - 2*d3*q, d1*p + d2*q, -1]
        ])
    
        return np.vstack((dg_ex, gyro_jacobian))  # final shape (6x6)    

    def f_6d(self, phi, theta, p, q, r, z_ddot):
        dt = self.dt
    
        # Update angles using gyro rates
        phi_next = phi + dt * ( p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta) )
        theta_next = theta + dt * ( q * np.cos(phi) - r * np.sin(phi) )
    
        # Other states remain unchanged
        p_next = p
        q_next = q
        r_next = r
        z_ddot_next = z_ddot
    
        return np.array([[phi_next], [theta_next], [p_next], [q_next], [r_next], [z_ddot_next]])
    
    def deltaf_6d(self, phi, theta, p, q, r, z_ddot):
        dt = self.dt
        sin_phi, cos_phi, tan_theta = np.sin(phi), np.cos(phi), np.tan(theta)
        sec_theta2 = 1.0 / (np.cos(theta)**2)
           
        F = np.array([
            [1 + dt*(q*cos_phi*tan_theta - r*sin_phi*tan_theta), dt*(q*sin_phi + r*cos_phi)*sec_theta2, dt, dt*sin_phi*tan_theta, dt*cos_phi*tan_theta, 0],
            [dt*(-q*sin_phi - r*cos_phi), 1, 0, dt*cos_phi, -dt*sin_phi, 0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,1]
        ])

        return F

