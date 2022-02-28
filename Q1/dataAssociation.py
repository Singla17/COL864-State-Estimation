import numpy as np
from Kalman_Filter import KalmanFilter 
from Agent import aeroplane 
import matplotlib.pyplot as plt

DELTA_T = 0.1
def dataAssociation(beliefs,observations):
    """
    Returns sorted listed of observations with respect to agent
    beliefs:list of n-tuples where n is number of agents(mean,covariance) containing predicted states after action update
    observations:list of np_array containing observations at a time step 
    """
    means=[x[0][:2] for x  in beliefs]
    for i in range(len(means)):
        mean=means[i]
        idx=min(range(i,len(observations)),key=lambda idx:np.sum(np.square(observations[idx]-mean)))
        observations[i],observations[idx]=observations[idx],observations[i]
    return observations

def multiagent_simulate(kf_filters,num_iters):
    """Simulates behviour of multiple agents using nearest neighbour based data association
    """    
    
    x_state =[[] for _ in range(len(kf_filters))]#[[x_a...t=0 ][x_b]]
    y_state =[[] for _ in range(len(kf_filters))]

    x_estimated = [[] for _ in range(len(kf_filters))]
    y_estimated = [[] for _ in range(len(kf_filters))]
    
    x_t=[filt.agent.getState() for filt in kf_filters]#[x_a,]
    
    x_cap_t = [filt.mean_belief for filt in kf_filters]
    fig, ax = plt.subplots()
    for i in range(len(kf_filters)):
        x_state[i].append(x_t[i][0][0])
        y_state[i].append(x_t[i][1][0])
        x_estimated[i].append(x_cap_t[i][0][0])
        y_estimated[i].append(x_cap_t[i][1][0])

    basic_arr = np.arange(0,num_iters,1)
    delta_vel_x = np.sin(basic_arr*DELTA_T)
    delta_vel_y = np.cos(basic_arr*DELTA_T)
    
    # ith iteration ke end me hummein X_i+1 ki beleif mil jaata hai using u_i,X^_i,z_i+1
    for i in range(num_iters):
        u_t = np.array([[delta_vel_x[i],delta_vel_y[i]]]).T #u_t
        # agent is at state X_i+1
        for filter_obj in kf_filters:
            filter_obj.agent.updateState(u_t)
        
        # action update done by kalman filter
        for filter_obj in kf_filters:
            filter_obj.mean_belief,filter_obj.covar_belief=filter_obj.action_update(u_t)
        
        # data association
        filter_beliefs=[(filter_obj.mean_belief,filter_obj.covar_belief) for filter_obj in kf_filters]    
        observations=[filter_obj.agent.get_observation() for filter_obj in kf_filters]
        
        observations=dataAssociation(filter_beliefs,observations)
        
        #measurement update
        for j,filter_obj in enumerate(kf_filters):
            filter_obj.mean_belief,filter_obj.covar_belief=filter_obj.measurement_update(filter_beliefs[j][0],filter_beliefs[j][1],observations[j])

        x_t=[filt.agent.getState() for filt in kf_filters]#[x_a,]
        
        x_cap_t = [filt.mean_belief for filt in kf_filters] 

        
        for j in range(len(kf_filters)):
            x_state[j].append(x_t[j][0][0])
            y_state[j].append(x_t[j][1][0])
            x_estimated[j].append(x_cap_t[j][0][0])
            y_estimated[j].append(x_cap_t[j][1][0])
     
    ax.set_title("Simulation")
    for j in range(len(kf_filters)):
        ax.plot(x_state[j], y_state[j], label = f"Actual Trajectory:{j}")
        
        ax.plot(x_estimated[j],y_estimated[j], label=f"Estimated Trajectory:{j}")
    
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    fig.canvas.draw()
    plt.legend()
    # plt.show()
    plt.savefig(f"figs/Q1/f_{len(kf_filters)}_agents.png")

if __name__ == '__main__':
    np.random.seed(0)
    
    init_state_a = np.array([[0,0,5,10]]).T
    A_t = np.array([[1,0,DELTA_T,0],[0,1,0,DELTA_T],[0,0,1,0],[0,0,0,1]])
    B_t = np.array([[0,0],[0,0],[1,0],[0,1]])
    C_t = np.array([[1,0,0,0],[0,1,0,0]])
    R_t_a = np.array([[1,0,0,0],[0,1,0,0],[0,0,1e-4,0],[0,0,0,1e-4]])
    Q_t_a= np.array([[100,0],[0,100]]) 
    
    mean_belief_0_a = np.array([[0,0,5,10]]).T
    covar_belief_0_a = np.array([[1e-4,0,0,0],[0,1e-4,0,0],[0,0,1e-4,0],[0,0,0,1e-4]])
    
    aero_obj_a = aeroplane(init_state_a,A_t,B_t,C_t,R_t_a,Q_t_a)
    estimator_a = KalmanFilter(aero_obj_a, mean_belief_0_a, covar_belief_0_a)


    init_state_b = np.array([[10,20,1,20]]).T
    R_t_b = np.array([[2,0,0,0],[0,2,0,0],[0,0,1e-4,0],[0,0,0,1e-4]])
    Q_t_b = np.array([[10,0],[0,10]]) 
    
    mean_belief_0_b = np.array([[10,20,1,21]]).T
    covar_belief_0_b = np.array([[1e-4,0,0,0],[0,1e-4,0,0],[0,0,1e-4,0],[0,0,0,1e-4]])
    
    aero_obj_b = aeroplane(init_state_b,A_t,B_t,C_t,R_t_b,Q_t_b)
    estimator_b = KalmanFilter(aero_obj_b, mean_belief_0_b, covar_belief_0_b)
    
    # init_state_c = np.array([[100,120,9,18]]).T
    # # A_t = np.array([[1,0,DELTA_T,0],[0,1,DELTA_T,0],[0,0,1,0],[0,0,0,1]])
    # # B_t = np.array([[0,0],[0,0],[1,0],[0,1]])
    # # C_t = np.array([[1,0,0,0],[0,1,0,0]])
    # R_t_c = np.array([[2,0,0,0],[0,2,0,0],[0,0,1e-4,0],[0,0,0,1e-4]])
    # Q_t_c = np.array([[10,0],[0,10]]) 
    
    # mean_belief_0_c = np.array([[100,120,9,18]]).T
    # covar_belief_0_c = np.array([[1e-4,0,0,0],[0,1e-4,0,0],[0,0,1e-4,0],[0,0,0,1e-4]])
    
    # aero_obj_c = aeroplane(init_state_c,A_t,B_t,C_t,R_t_c,Q_t_c)
    # estimator_c = KalmanFilter(aero_obj_c, mean_belief_0_c, covar_belief_0_c)
    
    # init_state_d = np.array([[100,20,50,0]]).T
    # # A_t = np.array([[1,0,DELTA_T,0],[0,1,DELTA_T,0],[0,0,1,0],[0,0,0,1]])
    # # /# B_t = np.array([[0,0],[0,0],[1,0],[0,1]])
    # # C_t = np.array([[1,0,0,0],[0,1,0,0]])
    # R_t_d = np.array([[5,0,0,0],[0,5,0,0],[0,0,1e-4,0],[0,0,0,1e-4]])
    # Q_t_d = np.array([[10,0],[0,10]]) 
    
    # mean_belief_0_d = np.array([[100,20,50,0]]).T
    # covar_belief_0_d = np.array([[1e-4,0,0,0],[0,1e-4,0,0],[0,0,1e-4,0],[0,0,0,1e-4]])
    
    # aero_obj_d= aeroplane(init_state_d,A_t,B_t,C_t,R_t_d,Q_t_d)
    # estimator_d = KalmanFilter(aero_obj_d, mean_belief_0_d, covar_belief_0_d)
    

    
    
    estimators=[estimator_a,estimator_b]
    multiagent_simulate(estimators,200)

    # pass
