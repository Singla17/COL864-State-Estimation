from  grid import Grid, visualise
import numpy as np
DIR_VECS=[[1,0],[0,1],[-1,0],[0,-1]]
class Robot:
    
    def __init__(self,init_state,r_max,grid):
        """
        Creates a robot with its initial true state and four noisy sensors in N,S,E,W dirs.Each sensor gives a discrete binary observation 
        whether  a wall is close (1) or far(0)
        init_state: 2x1 np array which denotes initial position of robot (x,y)
        r_max:int which denotes range of senor 
        grid:object of Grid type  in which robot is placed
        Action Model updates robots true state at each time step X_t---> X_t+1
        Sensor Model gives 4x1 numpy array  denoting observation from each sensor z_t 
        """
        self.state=init_state
        self.r_max=r_max
        self.grid=(grid)
        
    def updateState(self):
        """
        Choses a feasible action randomly and updates state of robot
        """
        x=self.state[0][0]
        y=self.state[1][0]
        possible_pos=[]
        for vec in DIR_VECS:
            cand_x=x+vec[0]
            cand_y=y+vec[1]
            if(self.grid.isfeasible((cand_x,cand_y))):
                possible_pos.append(np.array([cand_x,cand_y])[:,np.newaxis])
        print(possible_pos)
        if  possible_pos is None:
            print(f"Locked in Grid at pos ({x},{y}).Cannot Move") 
        self.state=possible_pos[(np.random.randint(0,len(possible_pos)))]


        
    def getObservation(self):
        pass


if __name__=="__main__":
    g=Grid(9,5,[(3,1),(3,2),(4,1),(4,2)])
    # visualise(g)
    init_state=np.array([2,1])[:,np.newaxis]
    robo=Robot(init_state,5,g)
    robo.updateState()
    print(robo.state)
    visualise(g)