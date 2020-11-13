import numpy as np
import math
import copy

class PolicyGradient:
    def __init__(self,mu,sigma,alpha_mu,alpha_sigma,gamma=0.95,n_rollout=10):
        self.alpha_mu,self.alpha_sigma = alpha_mu,alpha_sigma
        self.gamma = gamma
        self.n_rollout = n_rollout

        self.mu = mu
        self.sigma = sigma
        self.mu_history_ = copy.copy(self.mu)
        self.sigma_history_ = copy.copy(self.sigma)
        self.reward_history_ = np.array([0])

    def get_theta(self):
        pass

    def get_baseline(self):
        pass

    def train(self,theta,):
        pass

