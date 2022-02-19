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
        
        
    def updateBelief(self,u_t):
        
        mu_bar_t = np.matmul(self.agent.A_t,self.mean_belief) + np.matmul(self.agent.B_t,u_t)
        sigma_bar_t = np.matmul(np.matmul(self.agent.A_t,self.covar_belief),self.agent.A_t.T) + self.agent.R_t
        intermediate_var = np.matmul(np.matmul(self.agent.C_t,sigma_bar_t),self.agent.C_t.T) + self.agent.Q_t ## see Kalman Algo
        K_t = np.matmul(np.matmul(sigma_bar_t,self.agent.C_t.T),np.linalg.inv(intermediate_var))
        self.mean_belief = mu_bar_t + np.matmul(K_t,self.agent.get_observation()-np.matmul(self.agent.C_t,mu_bar_t))
        intermediate_var_2 = np.matmul(K_t,self.agent.C_t)
        intermediate_var_3 = np.identity(np.shape(intermediate_var_2)[0])-intermediate_var_2
        self.covar_belief = np.matmul(intermediate_var_3,sigma_bar_t)
        self.agent.updateState(u_t)
        
        return (self.mean_belief,self.covar_belief)
    
    def getBelief(self):
       
        return (self.mean_belief,self.covar_belief)
  
    
def simulate_filter(filter_obj,num_iters):
    pass
    
if __name__ == "__main__":
    
    np.random.seed(0)
    
    init_state = np.array([[0,0,10,10]]).T
    A_t = np.array([[1,0,DELTA_T,0],[0,1,DELTA_T,0],[0,0,1,0],[0,0,0,1]])
    B_t = np.array([[0,0],[0,0],[1,0],[0,1]])
    C_t = np.array([[1,0,0,0],[0,1,0,0]])
    R_t = np.array([[1,0,0,0],[0,1,0,0],[0,0,1e-4,0],[0,0,0,1e-4]])
    Q_t = np.array([[100,0],[0,100]]) 
    
    mean_belief_0 = np.array([[0,0,0,0]]).T
    covar_belief_0 = np.array([[1e-4,0,0,0],[0,1e-4,0,0],[0,0,0,1e-4],[0,0,0,1e-4]])
    
    aero_obj = aeroplane(init_state,A_t,B_t,C_t,R_t,Q_t)
    estimator = KalmanFilter(aero_obj, mean_belief_0, covar_belief_0)