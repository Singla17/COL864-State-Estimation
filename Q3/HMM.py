from hashlib import shake_128
import numpy as np
from Robot import Robot,DIR_VECS
from grid import Grid,visualise
def normalize(belief):
    summation=0
    for value in belief.values():
        summation+=value
    for key in belief.keys():
        belief[key]/=summation
    
class HMM:
    def __init__(self,robot):
        
        self.all_states=[]
        self.belief={}
        self.robot=robot
        self.grid=robot.grid
        # Populate all states with states from grid and generate initial belief from initial_state  
        for x in range(self.grid.length):
            for y in range(self.grid.breadth):
                pos=(x,y)
                self.all_states.append(pos)
                self.belief[pos]=0
        initial_state=robot.state
        self.belief[initial_state]=1

    def processModel(self,next_state,curr_state):
        """
        Returns P(X_t+1|X_t)
        """
        x,y=curr_state
        possible_pos=[]
        for vec in DIR_VECS:
            cand_x=x+vec[0]
            cand_y=y+vec[1]
            cand_pos=(cand_x,cand_y)
            if(self.grid.isfeasible(cand_pos)):
                possible_pos.append(cand_pos)
        if next_state in possible_pos:
            return 1/len(possible_pos)
        else :
            return 0
    def sensorModel(self,obs,pos):
        """
        Returns P(e_t+1|X_t+1)
        """
        dist_N=self.grid.obsDistance_N(pos)
        dist_S=self.grid.obsDistance_S(pos)
        dist_W=self.grid.obsDistance_W(pos)
        dist_E=self.grid.obsDistance_E(pos)
        Rmax=self.robot.r_max
        prob_N=1-((dist_N-1)/(Rmax-1)) if dist_N<Rmax else 0
        prob_S=1-((dist_S-1)/(Rmax-1)) if dist_S<Rmax else 0
        prob_E=1-((dist_E-1)/(Rmax-1)) if dist_E<Rmax else 0
        prob_W=1-((dist_W-1)/(Rmax-1)) if dist_W<Rmax else 0
        probs=(prob_N,prob_S,prob_E,prob_W)
        ans=1
        for evid,prob in zip(obs,probs):
            if evid==1:
                ans*=prob
            else:
                ans*=(1-prob)
        return ans


        
    def dynamicsUpdate(self):
        """
        Calculates and Returns P(X_t+1|e0,e1,...et) from P(X_t|e0,e1...et)
        """
        partial_belief={}
        for next_state in self.all_states:
            ans=0
            for state in self.all_states:
                ans+=self.processModel(next_state,state)*self.belief[state]
            partial_belief[next_state]=ans

        return partial_belief
    def measurementUpdate(self,partial_belief,next_observation):
        """
        Calculates and Returns P(X_t+1|e0,e1,...et,et+1) from P(X_t+1|e0,e1...et) and e_t+1
        """
        for next_state in self.all_states:
            self.belief[next_state]=self.sensorModel(next_observation,next_state)*partial_belief[next_state]
        normalize(self.belief)
        return self.belief
    def getStateEstimate(self):
        """
        Returns estimated state
        """
        estimated_state = max(self.belief, key= lambda x: self.belief[x])
        return estimated_state


if __name__ == '__main__':
    g=Grid(9,5,[(3,1),(3,2),(4,1),(4,2)])
    visualise(g)
    init_state=(1,3)
    robo=Robot(init_state,5,g)
    filt_obj=HMM(robo)
    for t in range(25):
        robo.updateState()
        observation=robo.getObservation()        
        part_belief=filt_obj.dynamicsUpdate()
        filt_obj.measurementUpdate(part_belief,observation)
    print(robo.path)
    print(filt_obj.getStateEstimate())

