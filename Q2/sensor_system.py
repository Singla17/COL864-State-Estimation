# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 12:44:45 2022

@author: Somanshu
"""

import math
import numpy as np
class sensors:
    
    def __init__(self,landmarks,radius,var_dt):
        """
        Returns a sensor system object
        landmarks:list of locations of landmarks in form of list of numpy 2x1 array
        radius: radius of the sensor as an int/float
        var_dt: variance of zero mean gaussian noise to be added in distance 
        """
        self.landmarks = landmarks
        self.radius = radius
        self.var_dt=var_dt
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
    
    def getObservation(self,location):
        """
        Returns tuple of landmark position and euclidean distance if in radius else None 
        """
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
            
            