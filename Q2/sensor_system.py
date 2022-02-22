# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 12:44:45 2022

@author: Somanshu
"""

import math
import numpy as np
class sensors:
    
    def __init__(self,landmarks,radius,var_dt):
        self.landmarks = landmarks
        self.radius = radius
        self.var_dt=var_dt
    def getRadius(self):
        return self.radius
    
    def getLandmarks(self):
        return self.landmarks
    
    def getLandmark(self,index):
        return self.landmarks[index]
    
    def getObservation(self,location):
        
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
            
            if dist<self.radius:
                if dist< nearest_dist:
                    nearest_dist = dist
                    sensor_index = i
                    
            if sensor_index == -1:
                return None
            else:
                nearest_dist+=np.random.normal(0,self.var_dt)
                return (self.landmarks[sensor_index], nearest_dist)
            
            