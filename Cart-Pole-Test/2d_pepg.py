import numpy as np
import matplotlib.pyplot as plt
import math
import gym
from rl_syspepg import rlsysPEPGAgent_reactive
import time
from math import pi


# pip3 install gym/
## Initialize the environment, agent, and hyperparameters
env = gym.make('Crazyflie2D-v0')
env._max_episode_steps=25000

alpha_mu = np.array([[0.2], [0.2]]) # mu Learning Rates
alpha_sigma = np.array([[0.01], [0.01]]) # sigma Learning Rates
agent = rlsysPEPGAgent_reactive(_alpha_mu=alpha_mu, _alpha_sigma=alpha_sigma, _gamma=0.95, _n_rollout=10)
# agent.mu_ = np.array([[30], [-1.5], [90], [3]])

agent.mu_ = np.array([[5], [10]])
agent.sigma_ = np.array([[1.0], [1.0]])


##### Reduce system down to pole parameters


## Initial figure setup
# There is something about the plotting that is slowing everything down
plt.ion()  # interactive on
fig = plt.figure()
plt.grid()
plt.xlim([-10,200])
plt.ylim([125,200])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title("Episode: {} Run: {}".format(0,0))
plt.show() 

# ============================
##          Episode 
# ============================
for k_ep in range(500): # k_episodes

    done = False # True when rollout complete
    mu_, sigma_ = agent.mu_, agent.sigma_
    r_cum = np.zeros(shape=(2*agent.n_rollout_,1))

    ## Show episode starting hyperparameters
    print("=============================================")
    print("=============================================")
    print("STARTING Episode # %d" %k_ep)
    print( 'Mu_RREV=%.3f \t Mu_gr=%.3f' %(mu_[0], mu_[1]))
    print( 'sig_RREV=%.3f \t sig_gr=%.3f' %(sigma_[0], sigma_[1]))
    print()
    

    ## Calc and display rollout parameters
    theta_rl, epsilon_rl = agent.get_theta()
    print( 'theta_rl = ')
    np.set_printoptions(precision=3, suppress=True)
    print(theta_rl[0,:], "--> RREV")
    print(theta_rl[1,:], "--> Gain Rate")


    # ============================
    ##          Run 
    # ============================
    k_run = 0
    while k_run < 2*agent.n_rollout_:

        state = env.reset()
        #time.sleep(1)
        R = 0
        
        # ============================
        ##          Rollout 
        # ============================
        done_run = False
        pitch_triggerd = False
        vz_initial = 3.0

        RREV_trigger = theta_rl[0,k_run]
        gain_rate = theta_rl[1,k_run]
        while (not done_run):
            if k_ep % 50 == 0 and k_run % 12 == 0:
                env.render()
                #time.sleep(0.01)

            d = env.ceiling_height - state[2]
            vz = state[3]
            RREV = vz/d
            if env.step_count % 10 == 0:
                #print("RREV = {:.2f}".format(RREV))
                pass
            if RREV > RREV_trigger and not pitch_triggerd:
                pitch_triggerd = True
                print("-"*10)

                pitch_rate = gain_rate*RREV
                print("Pitch Starts / RREV = {0:.3f} / pitch_rate = {1:.3f}".format(RREV,pitch_rate))
        
            if state[4] > pi/2:
                state,reward,done_run,info = env.step({"type":0})
            elif pitch_triggerd:
                state,reward,done_run,info = env.step({"type":4,"pitchrate":pitch_rate})        
                
            else:
                state,reward,done_run,info = env.step({"type":2,"vel_x":0.0,"vel_z":vz_initial})
            
            R += reward
        #time.sleep(10)
        
        
        r_cum[k_run] = R 
        print("reward = {:.4f}".format(R))
        ## Rollout Plotting
        if k_run % 10 == 0:
            plt.plot(k_ep,r_cum[k_run],marker = "_", color = "black", alpha = 0.5) 
            plt.title("Episode: {} Run: {}".format(k_ep, k_run+1))
            # If figure gets locked on fullscreen, press ctrl+f untill it's fixed (there's lag due to process running)
            plt.draw()
            plt.pause(0.0001)
            # fig.canvas.flush_events()

        k_run += 1

    print("Episode # %d training, average reward %.3f" %(k_ep, np.mean(r_cum)))
    agent.train(_theta=theta_rl, _reward=r_cum, _epsilon=epsilon_rl)


    plt.plot(k_ep, np.mean(r_cum),'r+')
    plt.draw()
    plt.pause(0.001)
    # fig.canvas.flush_events()

env.close()