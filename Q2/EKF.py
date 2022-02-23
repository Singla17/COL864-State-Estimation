import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path  
file = Path(__file__).resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory)) 

from Q1.Agent import aeroplane
from Q1.Kalman_Filter import KalmanFilter

DELTA_T = 1
NUM_SAMPLES_ELLIPSE = 5000

class ExtendedKalmanFilter():
    
    def __init__(self,agent,mean_belief_0,covar_belief_0):
        """Returns Extended Kalman Filter which can track agent
        agent: Agent object to be tracked needs to have getState function and specific covariance of noise 
        mean_belief:mean for initial state
        covar_belief: covariance for initial state  
        """
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
    
    def measurement_update(self,mu_bar_t,sigma_bar_t,z_t):
        """
        Returns updated belief(mean_belief, covar_belief) by introducing measurement z_t
        """
        H_t=self.agent.get_Ht()
        Q_t=self.agent.get_Qt()
        intermediate_var = np.matmul(np.matmul(H_t,sigma_bar_t),H_t.T) + Q_t ## see Kalman Algo
        K_t = np.matmul(np.matmul(sigma_bar_t,H_t.T),np.linalg.inv(intermediate_var))
        mean_belief = mu_bar_t + np.matmul(K_t,z_t-self.agent.h_t(mu_bar_t))
        intermediate_var_2 = np.matmul(K_t,H_t)
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





    