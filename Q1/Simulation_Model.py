# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 14:19:37 2022

@author: Somanshu
"""

import numpy as np
import matplotlib.pyplot as plt


DELTA_T = 0.1

class aeroplane():
    
    def __init__(self,initial_state,A_t,B_t,C_t,R_t,Q_t):
        self.state = initial_state
        self.A_t = A_t
        self.B_t = B_t
        self.C_t = C_t
        self.R_t = R_t
        self.Q_t = Q_t
        
    def getState(self):
        """
        Returns np_array
        -------
        self.state    
            current state of the model
        """
        return self.state
    
    def updateState(self,u_t):
        mu_noise = np.zeros((np.shape(self.R_t)[0],))
        next_state = np.matmul(self.A_t,self.state) + np.matmul(self.B_t,u_t) + np.expand_dims(np.random.multivariate_normal(mu_noise,self.R_t),1)
        self.state = next_state
        return next_state
    
    def get_observation(self):
        mu_noise = np.zeros((np.shape(self.Q_t)[0],))
        observation = np.matmul(self.C_t,self.state) + np.expand_dims(np.random.multivariate_normal(mu_noise,self.Q_t),1)
        return observation

        

def simulate(aero_obj,u_t,num_iters): 
    x_state = []
    y_state = []
    x_obs = []
    y_obs = []
    
    
    x_t = aero_obj.getState()
    z_t = aero_obj.get_observation()
    
    x_state.append(x_t[0][0])
    y_state.append(x_t[1][0])
    x_obs.append(z_t[0][0])
    y_obs.append(z_t[1][0])
    
    for i in range(num_iters):
        x_t = aero_obj.updateState(u_t)
        z_t = aero_obj.get_observation()
        
        x_state.append(x_t[0][0])
        y_state.append(x_t[1][0])
        x_obs.append(z_t[0][0])
        y_obs.append(z_t[1][0])
    
    plt.title("Simulation")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.plot(x_state, y_state, label = "Actual Trajectory")
    plt.plot(x_obs, y_obs, label = "Observed Trajectory")
    plt.legend()
    plt.show()

if __name__ =="__main__":
    
    np.random.seed(0)
    
    initial_state = np.array([[0,0,10,10]]).T
    A_t = np.array([[1,0,DELTA_T,0],[0,1,DELTA_T,0],[0,0,1,0],[0,0,0,1]])
    B_t = np.array([[0,0],[0,0],[1,0],[0,1]])
    C_t = np.array([[1,0,0,0],[0,1,0,0]])
    R_t = np.array([[1,0,0,0],[0,1,0,0],[0,0,1e-4,0],[0,0,0,1e-4]])
    Q_t = np.array([[100,0],[0,100]]) 
    u_t = np.array([[0,0]]).T
    
    aero_obj = aeroplane(initial_state,A_t,B_t,C_t,R_t,Q_t)
    
    simulate(aero_obj,u_t,200)
    
    
    