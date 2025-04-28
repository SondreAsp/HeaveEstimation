# -*- coding: utf-8 -*-
import numpy as np

# Import custom class
from EkfMatricesClass import EkfMatrices

class AttitudeClass:
    
    def __init__(self, dt, g, d, V, W, x_bar, X_bar):
        self.dt = dt
        self.g = g
        self.d = d
        self.V = V
        self.W = W
        self.x_bar = x_bar
        self.X_bar = X_bar
        self.ekfMatrix = EkfMatrices(dt, -g, d)
        
    def EKF_predict(self):
        # global x_bar, X_bar
        V = self.V
        x_bar = self.x_bar
        X_bar = self.X_bar

        phi, theta, p, q, r, z_ddot = x_bar.flatten()

        x_bar_kp1 = self.ekfMatrix.f_6d(phi, theta, p, q, r, z_ddot)

        F_k = self.ekfMatrix.deltaf_6d(phi, theta, p, q, r, z_ddot)

        X_bar_kp1 = F_k @ X_bar @ F_k.T + V  

        self.x_bar[:] = x_bar_kp1
        self.X_bar[:] = X_bar_kp1

    def EKF_correct(self, acc, gyro):
        # global x_bar, X_bar  
        W = self.W
        x_bar = self.x_bar
        X_bar = self.X_bar
        
        # Extract current state
        phi, theta, p, q, r, z_ddot = x_bar.flatten()
           
        # Updated measurement vector including gyro data
        y_k = np.array([acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2]]).reshape((6,1))
                   
        # Compute expected measurement
        y_bar = self.ekfMatrix.g_matrix_ex_6d(phi, theta, p, q, r, z_ddot)
           
        # Measurement Jacobian (6x6 including gyro)
        D = self.ekfMatrix.deltag_ex_6d(phi, theta, p, q, r, z_ddot)
           
        # Kalman gain computation
        S = D @ X_bar @ D.T + W
        K = X_bar @ D.T @ np.linalg.pinv(S)
                   
        # Correction step
        x_hat = x_bar + K @ (y_k - y_bar)
           
        # Covariance update
        I = np.eye(6)
        X_hat = (I - K @ D) @ X_bar @ (I - K @ D).T + K @ W @ K.T
           
        # State and covariance update
        x_bar[:] = x_hat
        X_bar[:] = X_hat
                
        return x_hat
        