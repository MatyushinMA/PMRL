import numpy as np
import numpy.random as npr
from scipy.special import softmax

class Activation(object):
    def __init__(self, skew, drop_level, drop_rate, ground, restore_level, restore_rate):
        self.skew = skew
        self.drop_level = drop_level
        self.drop_rate = drop_rate
        self.ground = ground
        self.restore_level = restore_level
        self.restore_rate = restore_rate
    
    def drop_restore(self, potentials, states):
        to_activate = []
        for i, pot in enumerate(potentials):
            if states[i] == -1:
                potentials[i] = self.restore_rate + pot
                if pot + self.restore_rate >= self.restore_level:
                    states[i] = 0
            elif states[i] == 1:
                potentials[i] = pot - self.drop_rate
                if pot - self.drop_rate <= self.drop_level:
                    states[i] = 0
            elif pot >= 1:
                states[i] = 1
            elif pot <= self.ground:
                states[i] = -1
            else:
                to_activate.append(i)
        return to_activate
    
    def freeze(self, potentials, states):
        for i, pot in enumerate(potentials):
            if pot >= 1:
                states[i] = 1
            elif pot <= self.ground:
                states[i] = -1
    
    def __call__(self, potentials, states):
        to_activate = self.drop_restore(potentials, states)
        activated = self.act(potentials[to_activate])
        activations = np.zeros(len(potentials))
        activations[to_activate] = activated
        self.freeze(potentials, states)
        return activations, to_activate
    
    def act(self, potentials):
        pass

class ReLU(Activation):
    def act(self, potentials):
        activations = np.zeros(len(potentials))
        for i, pot in enumerate(potentials):
            if pot <= 0:
                activations[i] = 0
            else:
                activations[i] = self.skew*pot
        return activations
        
Activations = {
    'relu' : ReLU
}

class Netline(object):
    def __init__(self, n, C,
                 num_inputs=1,
                 num_outputs=1,
                 skew=0.1,
                 activation='relu',
                 drop_level=0.7,
                 drop_rate=0.05,
                 ground=-0.1,
                 restore_level=0.1,
                 restore_rate=0.05,
                 no_backstimula=True,
                 reward_memory=100,
                 learning_rate=0.01):
        self.n = n
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.prev_activations = np.zeros(n)
        self.potentials = np.zeros(n)
        self.states = np.zeros(n)
        self.W = C.copy()
        self.no_backstimula = no_backstimula
        if no_backstimula:
            self.W -= np.diagonal(self.W)
        self.reward_memory = reward_memory
        self.learning_rate = learning_rate
        self.rewards = []
        self.proposal_shift = np.empty(0)
        self.activation = Activations[activation](skew, drop_level, drop_rate, ground, restore_level, restore_rate)
    
    def feed(self, inputs):
        if len(self.proposal_shift) > 0 and len(self.rewards) >= 2:
            probs = softmax(self.rewards)
            a = probs[-1]/probs[-2]
            odds = npr.random()
            if a >= 1 or odds <= a:
                self.W += self.proposal_shift
        self.proposal_shift = npr.normal(0, self.learning_rate, size=self.n**2).reshape((self.n, self.n))
        if self.no_backstimula:
            self.proposal_shift -= np.diagonal(self.proposal_shift)
        assert len(inputs) == self.num_inputs
        for i, inp in enumerate(inputs):
            if self.states[i] == 0:
                self.potentials[i] += inp
        new_activations, to_activate = self.activation(self.potentials, self.states)
        self.potentials[to_activate] = np.dot(self.W + self.proposal_shift, self.prev_activations)[to_activate] - new_activations[to_activate]
        self.prev_activations = new_activations
        self.rewards += [0]
        if len(self.rewards) > self.reward_memory:
            self.rewards = self.rewards[1:]
        return self.potentials[-self.num_outputs:]
    
    def reward(self, r):
        self.rewards[-1] += r
    
    def reset(self):
        self.prev_activations = np.zeros(self.n)
        self.potentials = np.zeros(self.n)
        self.states = np.zeros(self.n)
    
    def reset_rewards(self):
        self.proposal_shift = np.empty(0)
        self.rewards = []