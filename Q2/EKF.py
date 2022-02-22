import numpy as np
import matplotlib.pyplot as plt
import math
import sys,os
from sensor_system import sensors
from Agent import aeroplane

# from Agent import aeroplane
# from Agent import aeroplane
# from utils import confidence_ellipse

DELTA_T = 0.1
NUM_SAMPLES_ELLIPSE = 5000

class ExtendedKalmanFilter():
    
    def __init__(self,agent,sensor_system,mean_belief_0,covar_belief_0):
        """Returns Extended Kalman Filter which can track agent
        agent: Agent object to be tracked needs to have getState function and specific covariance of noise 
        sensor_system: Object of type sensors
        mean_belief:mean for initial state
        covar_belief: covariance for initial state  
        """
        self.agent = agent
        self.sen_sys=sensor_system
        self.mean_belief = mean_belief_0
        self.covar_belief = covar_belief_0
    def h_t(self,mu_bar_t):
        _,dist=self.sen_sys.getObservation(mu_bar_t[:2])
        ans=np.zeros((3,1))
        ans[0][0]=mu_bar_t[0][0]
        ans[1][0]=mu_bar_t[1][0]
        ans[2][0]=dist
        return ans 
    def get_Ht(self):
        
        pos=self.agent.getState()[:2]
        x_t=pos[0][0]
        y_t=pos[1][0]
        landmark,dist=self.sen_sys.getObservation(pos)
        ans=np.zeros((3,4))
        ans[0][0]=1
        ans[1][1]=1
        ans[2][0]=(x_t-landmark[0][0])/dist
        ans[2][1]=(y_t-landmark[1][0])/dist
        return ans

    def action_update(self,u_t):
        """
        Returns updated belief(mean_belief, covar_belief) by incorpoating action u_t
        """
        mu_bar_t = np.matmul(self.agent.A_t,self.mean_belief) + np.matmul(self.agent.B_t,u_t)
        sigma_bar_t = np.matmul(np.matmul(self.agent.A_t,self.covar_belief),self.agent.A_t.T) + self.agent.R_t   
        return (mu_bar_t,sigma_bar_t)
    def get_zt(self):
        pos=self.agent.getState()[:2]
        ans=self.agent.get_observation()
        _,dist=self.sen_sys.getObservation(pos)
        
        
        np.vstack((ans,dist))
        return ans
    def get_Qt(self):
        ans=np.zeros((3,3))
        ans[0][0]=self.agent.Q_t[0][0]
        ans[0][1]=self.agent.Q_t[0][1]
        ans[1][0]=self.agent.Q_t[1][0]
        ans[1][1]=self.agent.Q_t[1][1]
        ans[2][2]=self.sen_sys.var_dt
        return ans
    
    def measurement_update(self,mu_bar_t,sigma_bar_t):
        """
        Returns updated belief(mean_belief, covar_belief) by introducing measurement z_t
        """
        H_t=self.get_Ht()
        z_t=self.get_zt()
        Q_t=self.get_Qt()
        intermediate_var = np.matmul(np.matmul(H_t,sigma_bar_t),H_t.T) + Q_t ## see Kalman Algo
        K_t = np.matmul(np.matmul(sigma_bar_t,H_t.T),np.linalg.inv(intermediate_var))
        mean_belief = mu_bar_t + np.matmul(K_t,z_t-self.h_t(mu_bar_t))
        intermediate_var_2 = np.matmul(K_t,H_t)
        intermediate_var_3 = np.identity(np.shape(intermediate_var_2)[0])-intermediate_var_2
        covar_belief = np.matmul(intermediate_var_3,sigma_bar_t)
        return mean_belief, covar_belief    
    
    
    def updateBelief(self,u_t,var_dt,measurement_update=True):
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
  
if __name__ == "__main__":
    pass