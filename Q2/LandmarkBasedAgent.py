import numpy as np
import math
import sys
from pathlib import Path  
file = Path(__file__).resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory)) 

from Q1.Agent import aeroplane



import math
import numpy as np
DELTA_T=1
class modAeroplane(aeroplane):
    
    def __init__(self,initial_state,A_t,B_t,C_t,R_t,Q_t,landmarks,radius,var_dt):
        """
        Returns a sensor system object
        landmarks:list of locations of landmarks in form of list of numpy 2x1 array
        radius: radius of the sensor as an int/float
        var_dt: variance of zero mean gaussian noise to be added in distance 
        """
        super().__init__(initial_state,A_t,B_t,C_t,R_t,Q_t)
        self.landmarks = landmarks
        self.radius = radius
        self.var_dt=var_dt
    def h_t(self,mu_bar_t):
        """Observation function h() z_t=h(x_t)+del_t
        """
        landmark_loc,dist=self.getLandmarkObservation()
        dist=self.getEuclideanDistance(mu_bar_t[:2],landmark_loc)
        ans=np.zeros((3,1))
        ans[0][0]=mu_bar_t[0][0]
        ans[1][0]=mu_bar_t[1][0]
        ans[2][0]=dist
        return ans 
    def getEuclideanDistance(self,pos,landmark_loc):
        return math.sqrt(np.sum(np.square(pos-landmark_loc)))
    def get_Ht(self):
        """
        Returns Observation Jacobian if EKF is going to be used  
        """
        pos=self.getState()[:2]
        x_t=pos[0][0]
        y_t=pos[1][0]
        landmark,dist=self.getLandmarkObservation()
        ans=np.zeros((3,4))
        ans[0][0]=1
        ans[1][1]=1
        ans[2][0]=(x_t-landmark[0][0])/dist
        ans[2][1]=(y_t-landmark[1][0])/dist
        return ans
    def getObservation(self):
        ans=self.get_observation()
        pos=self.getLandmarkObservation()
        
        if pos:
            dist=pos[1]
            dist+np.random.normal(0,self.var_dt)
            ans=np.vstack((ans,dist))
        
        return ans
    def get_Qt(self):
        """
        Returns Q_t with landmark
        """
        ans=np.zeros((3,3))
        ans[0][0]=self.Q_t[0][0]
        ans[0][1]=self.Q_t[0][1]
        ans[1][0]=self.Q_t[1][0]
        ans[1][1]=self.Q_t[1][1]
        ans[2][2]=self.var_dt
        return ans
    
    
    def getRadius(self):
        """
        Returns radius of sensor
        """
        return self.radius
    
    def getLandmarks(self):
        """
        Returns list of landmarks in form of list of numpy 2x1 arrays (x,y)
        """
        return self.landmarks
    
    def getLandmark(self,index):
        """
        Returns specific landmark position as a numpy 2x1 array
        """
        return self.landmarks[index]
    
    def getLandmarkObservation(self):
        """
        Returns tuple of landmark position and euclidean distance from true state if in radius else None 
        """
        location=self.getState()[:2]
        nearest_dist = self.radius+1
        sensor_index = -1 
        
        for i in range(len(self.landmarks)):
            landmark= self.landmarks[i]
            x_landmark = landmark[0][0]
            y_landmark = landmark[1][0]
            x_loc = location[0][0]
            y_loc = location[1][0]
            
            dist_sq = (x_landmark-x_loc)**2 + (y_landmark-y_loc)**2
            dist = math.sqrt(dist_sq)
            
            if dist <= self.radius:
                if dist < nearest_dist:
                    nearest_dist = dist
                    sensor_index = i
                    
            if sensor_index == -1:
                return None
            else:
                
                return (self.landmarks[sensor_index], nearest_dist)
            
if __name__ == '__main__':
    np.random.seed(0)
    
    init_state = np.array([[31,0,10,10]]).T
    A_t = np.array([[1,0,DELTA_T,0],[0,1,DELTA_T,0],[0,0,1,0],[0,0,0,1]])
    B_t = np.array([[0,0],[0,0],[1,0],[0,1]])
    C_t = np.array([[1,0,0,0],[0,1,0,0]])
    R_t = np.array([[1,0,0,0],[0,1,0,0],[0,0,1e-4,0],[0,0,0,1e-4]])
    Q_t = np.array([[100,0],[0,100]]) 
    
    mean_belief_0 = np.array([[0,0,0,0]]).T
    covar_belief_0 = np.array([[1e-4,0,0,0],[0,1e-4,0,0],[0,0,1e-4,0],[0,0,0,1e-4]])
    landmarks=[np.array([0,0])[:,np.newaxis],np.array([100,100])[:,np.newaxis],np.array([-100,100])[:,np.newaxis],np.array([100,-100])[:,np.newaxis],np.array([-100,-100])[:,np.newaxis]]
    aero_obj = modAeroplane(init_state,A_t,B_t,C_t,R_t,Q_t,landmarks,30,1) 
    
