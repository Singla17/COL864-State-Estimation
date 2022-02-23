import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from pathlib import Path  
file = Path(__file__).resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory)) 

from Q1.Agent import aeroplane
from Q1.Kalman_Filter import KalmanFilter
from EKF import ExtendedKalmanFilter
from LandmarkBasedAgent import modAeroplane

DELTA_T=1
class MixedFilter():
    def __init__(self,agent,mean_belief,covar_belief):
        self.mean_belief = mean_belief
        self.covar_belief = covar_belief
        self.agent=agent
    
    def getBelief(self):
        """
        Returns mean and covariance(np_arrays)
        """
        return (self.mean_belief,self.covar_belief)
    def updateBelief(self,u_t):
        """Updates state's distribution by updating mean and covariance values
         u_t: Action at time t (np_array)"""
        #  z_t:np_array denoting measurement at time t"""
        kf=KalmanFilter(self.agent,self.mean_belief,self.covar_belief)
        mu_bar_t,sigma_bar_t = kf.action_update(u_t)
        z_t=self.agent.getObservation()
        if z_t.shape==(3,1):
            filt=ExtendedKalmanFilter(self.agent,mu_bar_t,sigma_bar_t)
        else:
            filt=KalmanFilter(self.agent,mu_bar_t,sigma_bar_t)
        print(type(filt),z_t,self.agent.getState()[:2])
        mu_cap_t,sigma_cap_t=filt.measurement_update(mu_bar_t,sigma_bar_t,z_t)
        self.mean_belief=mu_cap_t
        self.covar_belief= sigma_cap_t


def simulate_filter(filter_obj,num_iters):
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
     
    x_state.append(x_t[0][0])
    y_state.append(x_t[1][0])
    x_obs.append(z_t[0][0])
    y_obs.append(z_t[1][0])
    x_estimated.append(x_cap_t[0][0])
    y_estimated.append(x_cap_t[1][0])
    
    
    
    for i in range(num_iters):
        u_t = np.zeros((2,1))
        filter_obj.agent.updateState(u_t)
        filter_obj.updateBelief(u_t)
        
        x_t = filter_obj.agent.getState()
        z_t = filter_obj.agent.get_observation()
        x_cap_t = filter_obj.mean_belief
        
     
        
        x_state.append(x_t[0][0])
        y_state.append(x_t[1][0])
        x_obs.append(z_t[0][0])
        y_obs.append(z_t[1][0])
        x_estimated.append(x_cap_t[0][0])
        y_estimated.append(x_cap_t[1][0])
     
    ax.set_title("Simulation")
    
    ax.plot(x_state, y_state, label = "Actual Trajectory")
    # if observed_trajectory:
    ax.plot(x_obs, y_obs, label = "Observed Trajectory")
    ax.plot(x_estimated,y_estimated, label="Estimated Trajectory")
    
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    fig.canvas.draw()
    plt.legend()
    plt.show()
if __name__ == '__main__':
    init_state = np.array([[0,-5,4*math.cos(0.3),4*math.sin(0.3)]]).T
    A_t = np.array([[1,0,DELTA_T,0],[0,1,DELTA_T,0],[0,0,1,0],[0,0,0,1]])
    B_t = np.array([[0,0],[0,0],[1,0],[0,1]])
    C_t = np.array([[1,0,0,0],[0,1,0,0]])
    R_t = np.array([[1e-4,0,0,0],[0,1e-4,0,0],[0,0,1e-4,0],[0,0,0,1e-4]])
    Q_t = np.array([[100,0],[0,100]]) 
    
   
    landmarks=[np.array([0,0])[:,np.newaxis],np.array([100,100])[:,np.newaxis],np.array([-100,100])[:,np.newaxis],np.array([100,-100])[:,np.newaxis],np.array([-100,-100])[:,np.newaxis]]
    aero_obj = modAeroplane(init_state,A_t,B_t,C_t,R_t,Q_t,landmarks,30,1)


    mean_belief_0 = np.array([[-10,50,1,2]]).T
    covar_belief_0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    num_iters=5
    estimator=MixedFilter(aero_obj,mean_belief_0,covar_belief_0)
    simulate_filter(estimator,num_iters)
    