# -*- coding: utf-8 -*-
import numpy as np
class MotionMatrecies:
    
    def __init__(self, g):
        self.g = g
        
    @staticmethod
    def c_bv(psi, theta, phi): 
        R = np.array([
            [np.cos(psi)*np.cos(theta),
             np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi),
             np.cos(psi)*np.sin(theta)*np.cos(phi) + np.sin(psi)*np.sin(phi)],
        
            [np.sin(psi)*np.cos(theta),
             np.sin(psi)*np.sin(theta)*np.sin(phi) + np.cos(psi)*np.cos(phi),
             np.sin(psi)*np.sin(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi)],
        
            [-np.sin(theta),
             np.cos(theta)*np.sin(phi),
             np.cos(theta)*np.cos(phi)]
        ])
        
        return R
    
    @staticmethod
    def omega(psi, theta, phi, psi_dot, theta_dot, phi_dot):
        
        omega = np.array([
            [0.0,
             (phi_dot*np.sin(theta) - psi_dot),
             (phi_dot*np.sin(psi)*np.cos(theta) + theta_dot*np.cos(psi))],
            [-(phi_dot*np.sin(theta) - psi_dot),
             0.0,
             (-phi_dot*np.cos(psi)*np.cos(theta) + theta_dot*np.sin(psi))],
            [-(phi_dot*np.sin(psi)*np.cos(theta) + theta_dot*np.cos(psi)),
             (phi_dot*np.cos(psi)*np.cos(theta) - theta_dot*np.sin(psi)),
             0.0]
        ])
        
        return omega
    
    @staticmethod
    def omega_dot(psi, theta, phi, psi_ddot, theta_ddot, phi_ddot):
        
        omega_dot = np.array([
            [0.0,
             (phi_ddot*np.sin(theta) - psi_ddot),
             (phi_ddot*np.sin(psi)*np.cos(theta) + theta_ddot*np.cos(psi))],
            [-(phi_ddot*np.sin(theta) - psi_ddot),
             0.0,
             (-phi_ddot*np.cos(psi)*np.cos(theta) + theta_ddot*np.sin(psi))],
            [-(phi_ddot*np.sin(psi)*np.cos(theta) + theta_ddot*np.cos(psi)),
             (phi_ddot*np.cos(psi)*np.cos(theta) - theta_ddot*np.sin(psi)),
             0.0]
        ])
        
        return omega_dot
    
    def acc_z_global(self, acc, angle):
        g = self.g

        phi   = angle[0].item()
        theta = angle[1].item()
        psi   = 0
                
        # Compute rotation matrix from body to global frame.
        R = self.c_bv(psi, theta, phi)  # R is 3x3
    
        # Make sure accelerometer reading is a column vector.
        acc = np.array(acc).reshape((3, 1))
        
        # Define gravity in the NED (global) frame
        # Here, g is stored in self.g.
        g_inertial = np.array([[0], [0], [g]])
        
        # Convert gravity into the body frame.
        # Because R transforms from body to global,
        # the gravity in the body frame is given by R^T @ g_inertial.
        g_body = R.T @ g_inertial
    
        # Recover the actual acceleration at the sensor location in the body frame.
        # (The sensor measures specific force, i.e. a_P_body - g_body.)
        a_P_body = acc + g_body
    
        # Transform the sensor acceleration to the global frame.
        a_P_global = R @ a_P_body
    
        # Extract the z-component from the global acceleration.
        acc_z_global = a_P_global[2, 0]
        
        return acc_z_global

