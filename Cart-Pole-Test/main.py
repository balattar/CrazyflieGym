import numpy as np
import matplotlib.pyplot as plt
import math
import gym
from rl_syspepg import rlsysPEPGAgent_reactive
import time


# pip3 install gym/
## Initialize the environment, agent, and hyperparameters
env = gym.make('CartPole-v0')
env._max_episode_steps=25000

alpha_mu = np.array([[0.2], [0.2], [0.2], [0.2]]) # mu Learning Rates
alpha_sigma = np.array([[0.01], [0.01], [0.01], [0.01]]) # sigma Learning Rates
agent = rlsysPEPGAgent_reactive(_alpha_mu=alpha_mu, _alpha_sigma=alpha_sigma, _gamma=0.95, _n_rollout=4)
# agent.mu_ = np.array([[30], [-1.5], [90], [3]])

agent.mu_ = np.array([[15], [-5], [40], [5]])
agent.sigma_ = np.array([[2.0], [2.0], [2.0], [2.0]])


##### Reduce system down to pole parameters


## Initial figure setup
# There is something about the plotting that is slowing everything down
plt.ion()  # interactive on
fig = plt.figure()
plt.grid()
plt.xlim([-10,200])
plt.ylim([0,300])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title("Episode: {} Run: {}".format(0,0))
plt.show() 

# ============================
##          Episode 
# ============================
for k_ep in range(50): # k_episodes

    done = False # True when rollout complete
    mu_, sigma_ = agent.mu_, agent.sigma_
    r_cum = np.zeros(shape=(2*agent.n_rollout_,1))

    ## Show episode starting hyperparameters
    print("=============================================")
    print("=============================================")
    print("STARTING Episode # %d" %k_ep)
    print( 'Mu_kp=%.3f \t Mu_kd=%.3f \t Mu_tp=%.3f \t Mu_td=%.3f' %(mu_[0], mu_[1], mu_[2], mu_[3]))
    print( 'sig_kp=%.3f \t sig_kd=%.3f \t sig_tp=%.3f \t sig_td=%.3f' %(sigma_[0], sigma_[1], sigma_[2], sigma_[3]))
    print()
    

    ## Calc and display rollout parameters
    theta_rl, epsilon_rl = agent.get_theta()
    print( 'theta_rl = ')
    np.set_printoptions(precision=3, suppress=True)
    print(theta_rl[0,:], "--> kp")
    print(theta_rl[1,:], "--> kd")
    print(theta_rl[2,:], "--> ktheta_p")
    print(theta_rl[3,:], "--> ktheta_d")



    # ============================
    ##          Run 
    # ============================
    k_run = 0
    while k_run < 2*agent.n_rollout_:

        state = env.reset()
        R = 0
        
        # ============================
        ##          Rollout 
        # ============================
        for t in range(200):
            env.render()
            # print(observation)
            # print('%.3f %.3f %.3f %.3f' %(state[0], state[1], state[2]*180/math.pi, state[3]))
            action = theta_rl[:,k_run]
            state, reward, done, info = env.step(action)
            R += reward

            if done: # If cart goes outside bounds or pole falls too low
                # print("Episode finished after {} timesteps".format(t+1))
                break
        
        r_cum[k_run] = R 
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
    plt.pause(0.0001)
    # fig.canvas.flush_events()

env.close()