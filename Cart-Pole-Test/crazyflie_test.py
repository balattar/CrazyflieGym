import gym
import numpy as np
import time
from math import pi
env = gym.make('Crazyflie2D-v0')
state = env.reset()
done = False

RREV_trigger = 5.0
gain_rate = 10
vz_initial = 3
pitch_triggerd = False
while (not done):
    env.render()

    d = env.ceiling_height - state[2]
    vz = state[3]
    RREV = vz/d
    if env.step_count % 10 == 0:
        print("RREV = {:.2f}".format(RREV))
    if RREV > RREV_trigger and not pitch_triggerd:
        pitch_triggerd = True
        print("Pitch Starts",)
        pitch_rate = gain_rate*RREV
       
    
    #print(state[4])
    

    if state[4] > pi/2:
        state,reward,done,info = env.step({"type":0})
    elif pitch_triggerd:
        state,reward,done,info = env.step({"type":4,"pitchrate":pitch_rate})        
        
    else:
        state,reward,done,info = env.step({"type":2,"vel_x":0.0,"vel_z":vz_initial})
    #print(reward)



    #time.sleep(0.0001)

env.close()