import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
NO_OBS_DIST = 1e6

def visualise(grid,path=[]):
    """
    This functions renders the grid plot, where obstacles are shown in gray
    path : list of tuples (x,y) expressing path taken by robot
    """
        
    fig, ax = plt.subplots()
    plt.xlim(-0.5,grid.length-0.5)
    plt.ylim(-0.5,grid.breadth-0.5)
    # plt.axis(False)
    
    for i in range (grid.length):
        for j in range(grid.breadth):
            
            point = (i,j)
            
            if point in grid.obstacles:
                ax.add_patch(Rectangle((i-0.5,j-0.5),
                        1, 1,
                        fc ='gray', 
                        ec ='black',
                        lw = None))
            else:
                ax.add_patch(Rectangle((i-0.5,j-0.5),
                        1, 1,
                        fc ='none', 
                        ec ='black',
                        lw = None))
    circles=[]
    if path:
        for pos in path:
            x,y=pos
            circles.append(plt.Circle((x,y), 0.25))
        for circle in circles:
            ax.add_artist(circle)
        for i in range(len(path)-1):
            prev_pos,pos=path[i],path[i+1]
            x_pts=[prev_pos[0],pos[0]]
            y_pts=[prev_pos[1],pos[1]]
            plt.plot(x_pts,y_pts,color='b')

    

    plt.show()
        
        
def show_belief(grid,belief,t=0):
    fig, ax = plt.subplots()
    plt.style.use('grayscale')
    plt.xlim(-0.5,grid.length-0.5)
    plt.ylim(-0.5,grid.breadth-0.5)
    # plt.axis(False)
    log_likelihood=np.zeros((grid.breadth,grid.length))
    for pos,prob in belief.items():
        x,y=pos
        log_likelihood[y,x]=(1-prob)*255

    ax.imshow(log_likelihood)  

    plt.savefig(f"beliefs/{t}.png")

class Grid():
    """
    0-indexing i.e the first cell if the grid is (0,0) 
    obstacles is a list of tuples
    """
    
    def __init__(self,Length,Breadth,obstacles,add_walls=False):
        """
        Initialisation of grid and obstacles
        """
        self.length= Length
        self.breadth = Breadth
        self.obstacles = obstacles
        if add_walls:
            for x in range((self.length)):
                self.obstacles.append((x,0))
                self.obstacles.append((x,self.breadth-1))
            for y in range((self.breadth)):
                self.obstacles.append((0,y))
                self.obstacles.append((self.length-1,y))
                
        self.grid = np.zeros((self.length,self.breadth),dtype=int)
        
        for obstacle in self.obstacles:
            (x,y) = obstacle
            assert x< self.length and y<self.breadth, f" Obstacles ({x},{y}) out of grid range {self.length},{self.breadth}"
            self.grid[x][y] = 1
        
        self.grid = self.grid.T ## so as to emulate the real grid properly
            
    def obsDistance_N(self,position):
        """
        Return the distance of closest obstacle to the north of
        given position
        """
        x,y = position
        grid_column = self.grid[:,x]
        
        for i in range(y,len(grid_column)):
            if grid_column[i]!=0:
                return i-y
            
        return NO_OBS_DIST
    
    def obsDistance_S(self,position):
        """
        Return the distance of closest obstacle to the south of
        given position
        """
        x,y = position
        grid_column = self.grid[:,x]
        
        for i in range(y,-1,-1):
            if grid_column[i]!=0:
                return y-i
            
        return NO_OBS_DIST
    
    def obsDistance_E(self,position):
        """
        Return the distance of closest obstacle to the east of
        given position
        """
        x,y = position
        grid_row = self.grid[y]
        
        for i in range(x,len(grid_row)):
            if grid_row[i]!=0:
                return i-x
            
        return NO_OBS_DIST
    
    def obsDistance_W(self,position):
        """
        Return the distance of closest obstacle to the wet of
        given position
        """
        x,y = position
        grid_row = self.grid[y]
        
        for i in range(x,-1,-1):
            if grid_row[i]!=0:
                return x-i
            
        return NO_OBS_DIST
    
    def isfeasible(self,position):
        """
        Checks if the move to variable position is feasible
        """
        x,y=position
        if position not in self.obstacles and x>=0 and y>=0 and x<self.length and y<self.breadth:
            return True
        else:
            return False
    
        
if __name__ =="__main__":
    grid_obj = Grid(6,5,[(0,0),(4,4),(3,3),(3,4)])
    visualise(grid_obj)
    print(grid_obj.obsDistance_N((4,0)))
    print(grid_obj.obsDistance_S((1,3)))
    print(grid_obj.obsDistance_E((2,2)))
    print(grid_obj.obsDistance_W((3,0)))
                
                
        
            
        
    