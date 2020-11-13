import numpy as np
import copy

class rlsysPEPGAgent_reactive:

    def __init__(self,_alpha_mu, _alpha_sigma, _gamma=0.95, _n_rollout = 5):
        self.alpha_mu_, self.alpha_sigma_ = _alpha_mu, _alpha_sigma
        self.gamma_, self.n_rollout_ = _gamma, _n_rollout

        # Default values
        self.mu_ = np.array([[0], [0]])
        self.sigma_ = np.array([[0.001], [0.001]])

        self.mu_history_ = copy.copy(self.mu_)
        self.sigma_history_ = copy.copy(self.sigma_)
        self.reward_history_ = np.array([0])    
    
    def get_theta(self):
        zeros = np.zeros_like(self.mu_)
        # np.random.seed(0)
        epsilon = np.random.normal(zeros, self.sigma_, [zeros.size,self.n_rollout_])
        
        theta_plus = self.mu_ + epsilon
        theta_minus = self.mu_ - epsilon
        theta = np.append(theta_plus, theta_minus, axis=1)

        return theta, epsilon
    
    def get_baseline(self,_span):
        if self.reward_history_.size < _span:
            b = np.mean(self.reward_history_)
        else:
            b = np.mean(self.reward_history_[-_span:])

        return b

    def train(self, _theta, _reward, _epsilon):
        reward_plus = _reward[0:self.n_rollout_]
        reward_minus = _reward[self.n_rollout_:]
        epsilon = _epsilon
        b = self.get_baseline(_span=3)
        m_reward = 201 # max reward

        ## Decaying Learning Rate:
        # self.alpha_mu_ = self.alpha_mu_ * 0.9
        # self.alpha_sigma_ = self.alpha_sigma_ * 0.9

        T = epsilon
        S = (T**2 - self.sigma_**2)/self.sigma_


        ## If we maximize the episodes reward or the baseline equals the reward then
        # we get a divide by 0 error
        r_T = (reward_plus - reward_minus)/(2*m_reward - reward_plus - reward_minus)
        r_T = np.nan_to_num(r_T)

        r_S = ((reward_plus + reward_minus)/2 -b)/(m_reward - b)
        r_T = np.nan_to_num(r_T)

        self.mu_ = self.mu_ + self.alpha_mu_*np.dot(T,r_T)
        self.sigma_ = self.sigma_ + self.alpha_sigma_*np.dot(S,r_S)

        for k in range(self.sigma_.size): # If sigma oversteps negative then assume convergence
            if self.sigma_[k] <= 0:
                self.sigma_[k] = 0.001

        self.mu_history_ = np.append(self.mu_history_, self.mu_, axis=1)
        self.sigma_history_ = np.append(self.sigma_history_, self.sigma_, axis=1)
        self.reward_history_ = np.append(self.reward_history_, np.mean(_reward))