import gym
from gym import spaces, logger
from gym.utils import seeding

from math import sin,cos,pi,sqrt
import numpy as np

class Math3d:
    def __init__(self):
        pass
    def hat(self,vector):
        return np.array([[0,-vector[2],vector[1]],
                         [vector[2],0,-vector[0]]   
                         [-vector[1],vector[0],0] ]
        )
    def dehat(self,matrix):
        return np.array([(matrix[2,1]-matrix[1,2])/2],
                        [(matrix[0][2] - matrix[2][0])/2],
                        [(matrix[1][0] - matrix[0][1])/2]
        
        )

    
class Crazyflie2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.gravity = 9.81

        self.masscf = 28.7  # grams
        self.masslegs = 1.0 # grams
        self.mass_total = (self.masscf + self.masslegs)/1000  # kg
        self.length = 7.0/100.0 # meters
        self.inertia = 2.5*10**(-5) # get from 3d sim
        self.leg_length = 4.0/100.0 # meters
        self.time_elapsed = 0
        self.step_count = 0
        self.alpha = pi/4
        self.beta = pi/4

        self.ceiling_height = 1.5 # meters



        # self.force_mag = 10.0
        self.tau = 0.005  # seconds between state updates
        self.kinematics_integrator = 'euler'

        self.seed()
        self.viewer = None
        self.state = None

        self.action_space = spaces.Discrete(2) # not sure
        
        high = np.array([np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max],
                dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.steps_beyond_done = 0
        self.crashed = False
        self.landed = False


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



    def step(self, action):
        x,x_dot, z,z_dot, theta,theta_dot = self.state

        R = np.array([[cos(theta),-sin(theta)]
                    ,[sin(theta),cos(theta)]])

        self.time_elapsed += self.tau

        f_thrust = 0
        tau = 0

        if action["type"] == 2: # LinearVelocity 
            k_v = 0.7

            f_thrust = k_v*(action["vel_z"]-z_dot)  + self.gravity*self.mass_total
            #tau = k_omega*((theta % pi/4))
            tau = 0
            '''
            k_x = 0.2
            k_v = 0.7
            k_R = 0.02/2
            k_omega = 0.013/2

            # f = (-kx*ex - kv*ev + mg*e3 + mxd'')R*e3

            # Position error term
            e_x = np.zeros((2,1))

            # Velocity error term
            e_vx = action["vel_x"] - x_dot
            e_vz = action["vel_z"] - z_dot
            e_v = np.array([[e_vx],[e_vz]])

            # Gravity term
            mge3 = np.array([0,self.mass_total*self.gravity])
            mge3.shape = (2,1)


            f = (-k_x*e_x -k_v*e_v + mge3 )
            if f[1] < 0:
                f[1] = 0.01

            b3_d = f/np.linalg.norm(f)

            # eR = 0.5(RdTR - RTRd)V
            # eQ = Q - RTRdQd


            b1_d = np.zeros((3,1))
            b3_d_hat = Math3d().hat(vector=b3_d)
            b2_d = b3_d_hat.dot(b3_d) # multiply matrix vector
            b2_d = b2_d/math.linalg.norm(b2_d)
            b2_d_hat = Math3d().hat(b2_d)
            b1_d = b2_d_hat*b3_d

            R_d =enp.concatenate((b1_d,b2_d,b3_d),axis=1)

            R_dt = np.transpose(R_d)
            Rt = np.transpose(R)
            e_R = 0.5*dehat(R_dt.dot(R) - Rt.dot(R_d))

            # only have one? maybe doesnt work in SO(2)
            Q = np.array([theta_dot,0])
            e_Q = np.array([theta_dot,0])
            
            b3 = R[:,1]
            f_thrust = f.dot(b3)

            # T = -kr*eR - kQ*eQ + 
            #      QxJQ - J(hat(Q)RTRdQd - RTRdQ'd)
            tau_vec = -k_R*e_R 
            tau_scal = -k_omega*e_Q[0] + self.inertia*(theta_dot**2)

            tauA = tau_vec[0] + tau_scal
            tauB = tau_vec[1] + tau_scal

            tau = tauA
            '''
        elif action["type"] == 4: # Attitude Rate
            k_omega = 0.00005
            k_v = 3
            tau = k_omega*(action["pitchrate"]-theta_dot)
            f_thrust = k_v*tau/(self.length/2.0)



        if f_thrust > 2*self.mass_total*self.gravity:
            f_thrust = 2*self.mass_total*self.gravity

        if tau > 2*self.mass_total*self.gravity:
            tau = self.mass_total*self.gravity

        if self.crashed:
            f_thrust = 0
            tau = 0

            # correct for better impact
            if self.steps_beyond_done == 0:
                x_dot = 0
                z_dot = 0
                theta_dot = 0
                print("Crashed!")
            self.steps_beyond_done += 1
            z_acc = -self.gravity
            z_dot = z_dot + z_acc*self.tau
            z = z + z_dot*self.tau

        elif self.landed:
            print("Landed!")
            x_acc = 0
            x_dot = 0
            z_acc = 0
            z_dot = 0
            theta_acc = 0
            theta_dot = 0

        else:
            x_acc = -f_thrust*sin(theta)/self.mass_total
            x_dot = x_dot + x_acc*self.tau
            x = x + x_dot*self.tau

            z_acc = f_thrust*cos(theta)/self.mass_total - self.gravity
            z_dot = z_dot + z_acc*self.tau
            z = z + z_dot*self.tau

            theta_acc = tau/self.inertia
            theta_dot = theta_dot + theta_acc*self.tau
            theta = theta + theta_dot*self.tau

        self.state = (x,x_dot,z,z_dot,theta,theta_dot)

        self.z_right = z + self.length*sin(theta)/2.0 
        self.z_left =  z - self.length*sin(theta)/2.0 

        self.x_left = x - self.length*cos(theta)/2.0 
        self.x_right = x + self.length*cos(theta)/2.0 

        self.z_leg_right_joint = z + self.length*sin(theta)/4.0
        self.z_leg_left_joint = z - self.length*sin(theta)/4.0
        self.z_leg_right = self.z_leg_right_joint - self.leg_length*cos(self.alpha)*sin(theta)
        #print(self.z_leg_right)

        self.z_leg_left = self.z_leg_left_joint - self.leg_length*cos(self.beta)*sin(theta)

        self.x_leg_right_joint = x + self.length*cos(theta)/4.0
        self.x_leg_left_joint = x - self.length*cos(theta)/4.0
        self.x_leg_right = self.x_leg_right_joint + self.leg_length*sin(self.alpha)*cos(theta)
        self.x_leg_left = self.x_leg_left_joint - self.leg_length*sin(self.beta)*cos(theta)
        print("zll = {0:.3f} , zl = {1:.3f} zlj = {2:.3f} , z = {3:.3f} , zrj = {4:.3f} , zr = {5:.3f} , zrl = {6:.3f} , ".format(self.z_leg_left,self.z_left,self.z_leg_left_joint, z, self.z_leg_right_joint, self.z_right, self.z_leg_right,))
        print("{0:.2f} {1} {2}".format(theta,self.crashed,self.landed))

        done = bool(
            self.time_elapsed > 0.8
        )
        
        print(self.z_right >self.ceiling_height)
        if not self.landed:
            self.landed = bool(
                self.z_leg_left > self.ceiling_height
                or self.z_leg_right > self.ceiling_height
            )
        if not self.landed and not self.crashed:
            self.crashed = bool(
                z > self.ceiling_height or self.z_left > self.ceiling_height or self.z_right > self.ceiling_height)

        #print("{0:.3f} {1:.3f} {2:.3f} {3:.3f} {4:.3f} {4:.3f}".format(self.z_leg_left,self.z_left,self.z_leg_left_joint,self.z_leg_right_joint,self.z_right,self.z_leg_right))
        #print(self.landed,self.crashed)


        #r1 ~ height
        #r2 ~ orientation
        # r = r1+ r2

        r1 =  z/self.ceiling_height
        r2 =  1.5*(theta % pi/2)
        reward = r1+r2
        #print(r1,r2,reward)
        self.step_count += 1
        if self.step_count % 4 == 0:
            '''print("t = {0:.2f} \t [x,z,th] = [{1:.2f},{2:.2f},{3:.2f}] \t [xd,zd,thd] = [{4:.2f},{5:.2f},{6:.2f}] \t [f,tau] = [{7:.2f},{8:.6f}] \t".format(
                self.time_elapsed, self.state[0],self.state[2],self.state[4],self.state[1],self.state[3],self.state[5],f_thrust,tau
            ) )'''
        return np.array(self.state), reward, done, {}


    def reset(self):
        x = 0
        x_dot = 0

        z = 0
        z_dot = 0

        theta = 0
        theta_dot = 0

        self.state = (x,x_dot,z,z_dot,theta,theta_dot)

        self.z_right = z + self.length*sin(theta)/2.0 
        self.z_left =  z - self.length*sin(theta)/2.0 

        self.x_left = x - self.length*cos(theta)/2.0 
        self.x_right = x + self.length*cos(theta)/2.0 

        self.z_leg_right_joint = z + self.length*sin(theta)/4.0
        self.z_leg_left_joint = z - self.length*sin(theta)/4.0
        self.z_leg_right = self.z_leg_right_joint - self.leg_length*cos(self.alpha)

        self.z_leg_left = self.z_leg_left_joint - self.leg_length*cos(self.beta)

        self.x_leg_right_joint = x + self.length*cos(theta)/4.0
        self.x_leg_left_joint = x - self.length*cos(theta)/4.0
        self.x_leg_right = self.x_leg_right_joint + self.leg_length*sin(self.alpha)
        self.x_leg_left = self.x_leg_left_joint - self.leg_length*sin(self.beta)

        self.steps_beyond_done = 0
        self.step_count = 0
        self.landed = False
        self.crashed=False
        self.time_elapsed = 0
        return np.array(self.state)

    '''
    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound,bound,-bound,bound)

        if s is None: return None

        p1 = [-self.LINK_LENGTH_1 *
              cos(s[0]), self.LINK_LENGTH_1 * sin(s[0])]

        p2 = [p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]),
              p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1])]

        xys = np.array([[0,0], p1, p2])[:,::-1]
        thetas = [s[0]- pi/2, s[0]+s[1]-pi/2]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x,y),th,llen) in zip(xys, thetas, link_lengths):
            l,r,t,b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x,y))
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.add_attr(jtransform)
            link.set_color(0,.8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
    '''
    def render2(self,mode='human'):

        screen_width = 400
        screen_height = 600

        world_width = 2
        scale = screen_width/world_width
        scalez = screen_height/2  
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width,screen_height)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = 2 #self.x_threshold * 2
        scale = screen_width/world_width
        scalez = screen_height/2


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            base = rendering.Line((scale*self.x_left,scalez*self.z_left),(scale*self.x_right,scalez*self.z_right))
            base.set_color(0,0,0)
            self.basetrans = rendering.Transform()
            base.add_attr(self.basetrans)
            self.viewer.add_geom(base)

            self.leg_left = rendering.Line((scale*self.x_leg_left_joint,scalez*self.z_leg_left_joint),(scale*self.x_leg_left,scalez*self.z_leg_left))
            axleoffset = self.x_leg_left_joint
            self.leg_left.set_color(255,0,0)
            self.legltrans = rendering.Transform(translation = (scale*axleoffset,0),rotation=0.0)
            self.leg_left.add_attr(self.legltrans)
            self.leg_left.add_attr(self.basetrans)
            self.viewer.add_geom(self.leg_left)

            self.leg_right = rendering.Line((scale*self.x_leg_right_joint,scalez*self.z_leg_right_joint),(scale*self.x_leg_right,scalez*self.z_leg_right))
            self.leg_right.set_color(0,255,0)
            axleoffset = self.x_leg_right_joint
            self.legrtrans = rendering.Transform(translation=(scale*axleoffset,0),rotation=0.0)#translation)
            self.leg_right.add_attr(self.legrtrans)
            self.leg_right.add_attr(self.basetrans)
            self.viewer.add_geom(self.leg_right)

            self.ceiling = rendering.Line((0,scalez*self.ceiling_height),(screen_width,scalez*self.ceiling_height))
            self.ceiling.set_color(0,0,0)
            self.viewer.add_geom(self.ceiling)



            '''
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)

            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)

            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole'''

        if self.state is None:
            return None
        '''
        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state

        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])'''

        basex = self.state[0]*scale + screen_width/2.0
        basez = self.state[2]*scalez
        baseth = self.state[4]
        self.basetrans.set_translation(basex,basez)
        self.basetrans.set_rotation(baseth)

        #self.leg_right.start = (self.x_leg_right_joint,self.z_leg_right_joint)
        #self.leg_left.start = (self.x_leg_left_joint,self.z_leg_left_joint)
        #self.leg_right.end = (self.x_leg_right,self.z_leg_right)
        #self.leg_left.end = (self.x_leg_left,self.z_leg_left)

 



        #self.legltrans.set_translation(leglx,leglz)
        #self.legltrans.set_translation(legrx,legrz)





        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None