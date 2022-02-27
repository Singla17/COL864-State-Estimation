from  grid import Grid, visualise
import numpy as np
DIR_VECS=[[1,0],[0,1],[-1,0],[0,-1]]
class Robot:
    
    def __init__(self,init_state,r_max,grid):
        """
        Creates a robot with its initial true state and four noisy sensors in N,S,E,W dirs.Each sensor gives a discrete binary observation 
        whether  a wall is close (1) or far(0)
        init_state: Tuple which denotes initial position of robot (x,y)
        r_max:int which denotes range of senor 
        grid:object of Grid type  in which robot is placed
        Action Model updates robots true state at each time step X_t---> X_t+1
        Sensor Model gives  Tuple  denoting observation from each sensor z_t 
        """
        self.state=init_state
        self.r_max=r_max
        self.grid=(grid)
        self.path=[init_state]
        
    def updateState(self):
        """
        Choses a feasible action randomly and updates state of robot
        """
        x=self.state[0]
        y=self.state[1]
        possible_pos=[]
        for vec in DIR_VECS:
            cand_x=x+vec[0]
            cand_y=y+vec[1]
            cand_pos=(cand_x,cand_y)
            if(self.grid.isfeasible(cand_pos) ):
                possible_pos.append(cand_pos)
        # print(possible_pos)
        if  possible_pos is None:
            print(f"Locked in Grid at pos ({x},{y}).Cannot Move") 
        self.state=possible_pos[(np.random.randint(0,len(possible_pos)))]
        self.path.append(self.state)

        
    def getObservation(self):
        """
        Returns a 4 length tuple denoting observation from each sensor
        Each sensor gives a discrete binary observation {dn,ds,de,dw}
        whether  a wall is close (1) or far(0) 
        """
        pos=self.state
        # print(pos)
        dist_N=self.grid.obsDistance_N(pos)
        dist_S=self.grid.obsDistance_S(pos)
        dist_W=self.grid.obsDistance_W(pos)
        dist_E=self.grid.obsDistance_E(pos)
        Rmax=self.r_max
        if Rmax==1:
            prob_N=1 if dist_N==1 else 0
            dn=np.random.binomial(1,prob_N)
            prob_S=1 if dist_S==1  else 0
            ds=np.random.binomial(1,prob_S)
            prob_E=1 if dist_E==1  else 0
            de=np.random.binomial(1,prob_E)
            prob_W=1 if dist_W==1 else 0
            dw=np.random.binomial(1,prob_W)
            
        else:
            prob_N=1-((dist_N-1)/(Rmax-1)) if dist_N<Rmax and Rmax !=1 else 0
            dn=np.random.binomial(1,prob_N)
            prob_S=1-((dist_S-1)/(Rmax-1)) if dist_S<Rmax and Rmax !=1 else 0
            ds=np.random.binomial(1,prob_S)
            prob_E=1-((dist_E-1)/(Rmax-1)) if dist_E<Rmax and Rmax !=1 else 0
            de=np.random.binomial(1,prob_E)
            prob_W=1-((dist_W-1)/(Rmax-1)) if dist_W<Rmax and Rmax !=1 else 0
            dw=np.random.binomial(1,prob_W)
            # print(prob_N,prob_S,prob_E,prob_W)
        return (dn,ds,de,dw)

    
if __name__=="__main__":
    g=Grid(9,5,[(3,1),(3,2),(4,1),(4,2)])
    # visualise(g)
    init_state=(1,3)
    robo=Robot(init_state,5,g)
    robo.updateState()
    print(robo.path)
    print(robo.state)
    print(robo.getObservation())
    visualise(g)