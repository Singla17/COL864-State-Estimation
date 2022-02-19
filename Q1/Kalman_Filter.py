# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 15:20:00 2022

@author: Somanshu
"""

import numpy as np
import matplotlib.pyplot as plt
from Simulation_Model import aeroplane

DELTA_T = 0.1

class KalmanFilter():
    
    def __init__(self,agent,mean_belief_0,covar_belief_0):
        
        self.agent = agent
        self.mean_belief = mean_belief_0
        self.covar_belief = covar_belief_0
        
        
    def updateBelief(self,u_t,measurement_update=True):
        
        mu_bar_t = np.matmul(self.agent.A_t,self.mean_belief) + np.matmul(self.agent.B_t,u_t)
        sigma_bar_t = np.matmul(np.matmul(self.agent.A_t,self.covar_belief),self.agent.A_t.T) + self.agent.R_t
        
        if measurement_update:
            intermediate_var = np.matmul(np.matmul(self.agent.C_t,sigma_bar_t),self.agent.C_t.T) + self.agent.Q_t ## see Kalman Algo
            K_t = np.matmul(np.matmul(sigma_bar_t,self.agent.C_t.T),np.linalg.inv(intermediate_var))
            self.mean_belief = mu_bar_t + np.matmul(K_t,self.agent.get_observation()-np.matmul(self.agent.C_t,mu_bar_t))
            intermediate_var_2 = np.matmul(K_t,self.agent.C_t)
            intermediate_var_3 = np.identity(np.shape(intermediate_var_2)[0])-intermediate_var_2
            self.covar_belief = np.matmul(intermediate_var_3,sigma_bar_t)
            
        else:
            self.mean_belief = mu_bar_t
            self.covar_belief_t = sigma_bar_t
        
        self.agent.updateState(u_t)
        
        return (self.mean_belief,self.covar_belief)
    
    def getBelief(self):
       
        return (self.mean_belief,self.covar_belief)
  
    
def simulate_filter(filter_obj,num_iters,uncertainity_ellipse=False,loss_locs=[],loss_durations=[]):
    x_state = []
    y_state = []
    x_obs = []
    y_obs = []
    x_estimated = []
    y_estimated = []
    
    x_t = filter_obj.agent.getState()
    z_t = filter_obj.agent.get_observation()
    x_cap_t = np.random.multivariate_normal(np.squeeze(filter_obj.mean_belief),filter_obj.covar_belief)
    
    """
    if uncertainity_ellipse:
        x_ellipse, y_ellipse = np.random.multivariate_normal(filter_obj.mean_belief, filter_obj.covar_belief, 5000).T
        
    """  
    x_state.append(x_t[0][0])
    y_state.append(x_t[1][0])
    x_obs.append(z_t[0][0])
    y_obs.append(z_t[1][0])
    x_estimated.append(x_cap_t[0])
    y_estimated.append(x_cap_t[1])
    
    
    basic_arr = np.arange(0,num_iters,1)
    delta_vel_x = np.sin(basic_arr)
    delta_vel_y = np.cos(basic_arr)
    
    
    absence_of_observations = []
        
    if len(loss_locs)!=0:
        for i in range(len(loss_locs)): 
            duration = loss_durations[i]
            location_of_loss = loss_locs[i]
            
            for j in range(duration):
                absence_of_observations.append(location_of_loss+j)
      
    for i in range(num_iters):
        u_t = np.array([[delta_vel_x[i],delta_vel_y[i]]]).T
        filter_obj.updateBelief(u_t,not (i in absence_of_observations))
        
        x_t = filter_obj.agent.getState()
        z_t = filter_obj.agent.get_observation()
        x_cap_t = np.random.multivariate_normal(np.squeeze(filter_obj.mean_belief),filter_obj.covar_belief)
        
        x_state.append(x_t[0][0])
        y_state.append(x_t[1][0])
        x_obs.append(z_t[0][0])
        y_obs.append(z_t[1][0])
        x_estimated.append(x_cap_t[0])
        y_estimated.append(x_cap_t[1])
        
    plt.title("Simulation")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.plot(x_state, y_state, label = "Actual Trajectory")
    plt.plot(x_obs, y_obs, label = "Observed Trajectory")
    plt.plot(x_estimated,y_estimated, label="Estimated Trajectory")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    
    np.random.seed(0)
    
    init_state = np.array([[0,0,10,10]]).T
    A_t = np.array([[1,0,DELTA_T,0],[0,1,DELTA_T,0],[0,0,1,0],[0,0,0,1]])
    B_t = np.array([[0,0],[0,0],[1,0],[0,1]])
    C_t = np.array([[1,0,0,0],[0,1,0,0]])
    R_t = np.array([[1,0,0,0],[0,1,0,0],[0,0,1e-4,0],[0,0,0,1e-4]])
    Q_t = np.array([[100,0],[0,100]]) 
    
    mean_belief_0 = np.array([[0,0,0,0]]).T
    covar_belief_0 = np.array([[1e-4,0,0,0],[0,1e-4,0,0],[0,0,1e-4,0],[0,0,0,1e-4]])
    
    aero_obj = aeroplane(init_state,A_t,B_t,C_t,R_t,Q_t)
    estimator = KalmanFilter(aero_obj, mean_belief_0, covar_belief_0)
    simulate_filter(estimator,200,loss_locs=[10,60],loss_durations=[20,20])
    simulate_filter(estimator,200,loss_locs=[],loss_durations=[])