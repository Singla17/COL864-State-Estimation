# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 15:20:00 2022

@author: Somanshu
"""

import numpy as np
import matplotlib.pyplot as plt


DELTA_T = 0.1
NUM_SAMPLES_ELLIPSE = 5000

class KalmanFilter():
    
    def __init__(self,agent,mean_belief_0,covar_belief_0):
        
        self.agent = agent
        self.mean_belief = mean_belief_0
        self.covar_belief = covar_belief_0
        
    def action_update(self,u_t):
        """
        Returns updated belief(mean_belief, covar_belief) by incorpoating action u_t
        """
        mu_bar_t = np.matmul(self.agent.A_t,self.mean_belief) + np.matmul(self.agent.B_t,u_t)
        sigma_bar_t = np.matmul(np.matmul(self.agent.A_t,self.covar_belief),self.agent.A_t.T) + self.agent.R_t   
        return (mu_bar_t,sigma_bar_t)
    
    def measurement_update(self,mu_bar_t,sigma_bar_t,z_t=None):
        """
        Returns updated belief(mean_belief, covar_belief) by introducing measurement z_t
        """
        if z_t is None:
            z_t=self.agent.get_observation()
        intermediate_var = np.matmul(np.matmul(self.agent.C_t,sigma_bar_t),self.agent.C_t.T) + self.agent.Q_t ## see Kalman Algo
        K_t = np.matmul(np.matmul(sigma_bar_t,self.agent.C_t.T),np.linalg.inv(intermediate_var))
        mean_belief = mu_bar_t + np.matmul(K_t,z_t-np.matmul(self.agent.C_t,mu_bar_t))
        intermediate_var_2 = np.matmul(K_t,self.agent.C_t)
        intermediate_var_3 = np.identity(np.shape(intermediate_var_2)[0])-intermediate_var_2
        covar_belief = np.matmul(intermediate_var_3,sigma_bar_t)
        return mean_belief, covar_belief    
    
    
    def updateBelief(self,u_t,measurement_update=True):
        """
        Updates state's distribution by updating mean and covariance values
        u_t: Action at time t (np_array)
        z_t:np_array denoting measurement at time t
        measurement_update:boolean (tells if Measurement is given or not)
        """
        
        mu_bar_t,sigma_bar_t =self.action_update(u_t)
        if measurement_update:
            self.mean_belief,self.covar_belief=self.measurement_update(mu_bar_t,sigma_bar_t)   
        else:
            self.mean_belief = mu_bar_t
            self.covar_belief_t = sigma_bar_t
        
        
        
        return (self.mean_belief,self.covar_belief)
    
    def getBelief(self):
        """
        Returns mean and covariance(np_arrays)
        """
        return (self.mean_belief,self.covar_belief)
  
    
def simulate_filter(filter_obj,num_iters,uncertainity_ellipse=False,observed_trajectory=False,loss_locs=[],loss_durations=[]):
    """
    Simulates Kalman Filter for a given agent with initial state
    """
    x_state = []
    y_state = []
    x_obs = []
    y_obs = []
    x_estimated = []
    y_estimated = []
    
    x_t = filter_obj.agent.getState()
    z_t = filter_obj.agent.get_observation()
    x_cap_t = filter_obj.mean_belief
    
    fig, ax = plt.subplots()
    
    if uncertainity_ellipse:
        x_ellipse, y_ellipse,_,_ = np.random.multivariate_normal(np.squeeze(filter_obj.mean_belief), filter_obj.covar_belief, NUM_SAMPLES_ELLIPSE).T
        confidence_ellipse(x_ellipse, y_ellipse, ax, edgecolor='yellow')
     
    x_state.append(x_t[0][0])
    y_state.append(x_t[1][0])
    x_obs.append(z_t[0][0])
    y_obs.append(z_t[1][0])
    x_estimated.append(x_cap_t[0][0])
    y_estimated.append(x_cap_t[1][0])
    
    
    basic_arr = np.arange(0,num_iters,1)
    delta_vel_x = np.sin(basic_arr*DELTA_T)
    delta_vel_y = np.cos(basic_arr*DELTA_T)
    
    
    absence_of_observations = []
        
    if len(loss_locs)!=0:
        for i in range(len(loss_locs)): 
            duration = loss_durations[i]
            location_of_loss = loss_locs[i]
            
            for j in range(duration):
                absence_of_observations.append(location_of_loss+j)
      
    for i in range(num_iters):
        u_t = np.array([[delta_vel_x[i],delta_vel_y[i]]]).T
        filter_obj.agent.updateState(u_t)
        filter_obj.updateBelief(u_t,not (i in absence_of_observations))
        
        x_t = filter_obj.agent.getState()
        z_t = filter_obj.agent.get_observation()
        x_cap_t = filter_obj.mean_belief
        
        if uncertainity_ellipse:
            x_ellipse, y_ellipse,_,_ = np.random.multivariate_normal(np.squeeze(filter_obj.mean_belief), filter_obj.covar_belief, NUM_SAMPLES_ELLIPSE).T
            confidence_ellipse(x_ellipse, y_ellipse, ax, edgecolor='yellow')
        
        x_state.append(x_t[0][0])
        y_state.append(x_t[1][0])
        x_obs.append(z_t[0][0])
        y_obs.append(z_t[1][0])
        x_estimated.append(x_cap_t[0][0])
        y_estimated.append(x_cap_t[1][0])
     
    ax.set_title("Simulation")
    
    ax.plot(x_state, y_state, label = "Actual Trajectory")
    if observed_trajectory:
        ax.plot(x_obs, y_obs, label = "Observed Trajectory")
    ax.plot(x_estimated,y_estimated, label="Estimated Trajectory")
    
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    fig.canvas.draw()
    plt.legend()
    plt.savefig("figs/Q1/c_obs.png")
    plt.show()
    
def simulate_filter_vel(filter_obj,num_iters,uncertainity_ellipse=False,observed_trajectory=False,loss_locs=[],loss_durations=[]):
    """
    Simulates Kalman Filter for a given agent with initial state
    """
    x_state = []
    y_state = []
    x_obs = []
    y_obs = []
    x_estimated = []
    y_estimated = []
    
    x_t = filter_obj.agent.getState()
    z_t = filter_obj.agent.get_observation()
    x_cap_t = filter_obj.mean_belief
    
    fig, ax = plt.subplots()
    
    if uncertainity_ellipse:
        x_ellipse, y_ellipse,_,_ = np.random.multivariate_normal(np.squeeze(filter_obj.mean_belief), filter_obj.covar_belief, NUM_SAMPLES_ELLIPSE).T
        confidence_ellipse(x_ellipse, y_ellipse, ax, edgecolor='yellow')
     
    x_state.append(x_t[2][0])
    y_state.append(x_t[3][0])
    x_obs.append(z_t[0][0])
    y_obs.append(z_t[1][0])
    x_estimated.append(x_cap_t[2][0])
    y_estimated.append(x_cap_t[3][0])
    
    basic_arr = np.arange(0,num_iters,1)
    delta_vel_x = np.sin(basic_arr*DELTA_T)
    delta_vel_y = np.cos(basic_arr*DELTA_T)
    
    
    
    absence_of_observations = []
        
    if len(loss_locs)!=0:
        for i in range(len(loss_locs)): 
            duration = loss_durations[i]
            location_of_loss = loss_locs[i]
            
            for j in range(duration):
                absence_of_observations.append(location_of_loss+j)
      
    for i in range(num_iters):
        u_t = np.array([[delta_vel_x[i],delta_vel_y[i]]]).T
        filter_obj.agent.updateState(u_t)
        filter_obj.updateBelief(u_t,not (i in absence_of_observations))
        
        x_t = filter_obj.agent.getState()
        z_t = filter_obj.agent.get_observation()
        x_cap_t = filter_obj.mean_belief
        
        if uncertainity_ellipse:
            x_ellipse, y_ellipse,_,_ = np.random.multivariate_normal(np.squeeze(filter_obj.mean_belief), filter_obj.covar_belief, NUM_SAMPLES_ELLIPSE).T
            confidence_ellipse(x_ellipse, y_ellipse, ax, edgecolor='yellow')
        
        x_state.append(x_t[2][0])
        y_state.append(x_t[3][0])
        x_obs.append(z_t[0][0])
        y_obs.append(z_t[1][0])
        x_estimated.append(x_cap_t[2][0])
        y_estimated.append(x_cap_t[3][0])
     
    ax.set_title("Simulation")
    
    ax.plot(x_state, y_state, label = "Actual Velocity")
    if observed_trajectory:
        ax.plot(x_obs, y_obs, label = "Observed Trajectory")
    ax.plot(x_estimated,y_estimated, label="Estimated Velocity")
    
    plt.xlabel("Velocity along x-axis")
    plt.ylabel("Velocity along y-axis")
    fig.canvas.draw()
    plt.legend()
    plt.savefig("figs/Q1/e_equal_init.png")
    plt.show()
if __name__ == "__main__":
    from Agent import aeroplane    
    from utils import confidence_ellipse    
    np.random.seed(0)
    
    init_state = np.array([[0,0,10,10]]).T
    A_t = np.array([[1,0,DELTA_T,0],[0,1,0,DELTA_T],[0,0,1,0],[0,0,0,1]])
    B_t = np.array([[0,0],[0,0],[1,0],[0,1]])
    C_t = np.array([[1,0,0,0],[0,1,0,0]])
    R_t = np.array([[1,0,0,0],[0,1,0,0],[0,0,1e-4,0],[0,0,0,1e-4]])
    Q_t = np.array([[100,0],[0,100]]) 
    
    mean_belief_0 = np.array([[0,0,10,10]]).T
    covar_belief_0 = np.array([[1e-4,0,0,0],[0,1e-4,0,0],[0,0,1e-4,0],[0,0,0,1e-4]])
    
    aero_obj = aeroplane(init_state,A_t,B_t,C_t,R_t,Q_t)
    estimator = KalmanFilter(aero_obj, mean_belief_0, covar_belief_0)
    simulate_filter(estimator,200,uncertainity_ellipse=False,observed_trajectory=True,loss_locs=[],loss_durations=[])
    # simulate_filter_vel(estimator,200,uncertainity_ellipse=False,observed_trajectory=False,loss_locs=[],loss_durations=[])
