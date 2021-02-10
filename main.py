
# Library Imports
import gym

from collections import deque
import random
import numpy as np

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


##############################################################################
# Class for Defining the Deep Q-Network Agent
##############################################################################
class DQN_Agent:
    def __init__(self,env):
        
        #Hyperparams.
        self.env=env
        self.gamma=0.99
        self.epsilon = 1
        self.epsilon_decay = 0.04
        self.epsilon_min=0.01
        self.learning_rate=0.001
        
        self.replay_memory=deque(maxlen=20000)
    
        self.n_episodes=400
        self.len_episode=200
        self.buffer_size=32
        
        self.train_network=self.Policy_Network()
        self.target_network=self.Policy_Network()
        self.target_network.set_weights(self.train_network.get_weights())

##############################################################################
    def Policy_Network(self):
        """ Creates a dense network to estimate a policy """
        
        model = models.Sequential()
        state_shape = self.env.observation_space.shape

        model.add(Dense(16, activation='relu', input_shape=state_shape))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

##############################################################################
    def epsgreedyaction(self,state):
        """ Agent picks an action based on Epsilon-Greedy Policy """
        
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            action=np.argmax(self.train_network.predict(state)[0])

        return action

##############################################################################
    def train_buffer(self):
        """ Dataset based optimization using Tensorflow """
     
        if len(self.replay_memory) < self.buffer_size:
            return

        samples = random.sample(self.replay_memory,self.buffer_size)

        states = []
        new_states=[]
        for sample in samples:
            state, action, reward, new_state, done = sample
            states.append(state)
            new_states.append(new_state)

        states = np.array(states)
        new_states = np.array(new_states)
        
        states = states.reshape(self.buffer_size, 2)
        new_states = new_states.reshape(self.buffer_size, 2)

        targets = self.train_network.predict(states)
        new_state_targets=self.target_network.predict(new_states)

        for i,sample in enumerate(samples):
            state, action, reward, new_state, done = sample
            target = targets[i]
            if done:
                target[action] = reward
            else:
                Q_future = max(new_state_targets[i])
                target[action] = reward + Q_future * self.gamma
        
        self.train_network.fit(states, targets, epochs=10, verbose=0)


##############################################################################
    def run(self,current_state,eps):
        """ Generates Episodic Data """
        
        sum_reward = 0
        
        for i in range(self.len_episode):
            bestAction = self.epsgreedyaction(current_state)

            # Render the env. at the last iteration
            if eps % self.n_episodes==0:
                try:
                    env.render()
                except:
                    pass

            new_state, reward, done, _ = env.step(bestAction)
            new_state = new_state.reshape(1, 2)
                
            # Reward Engineering
            if new_state[0][0] >= 0.5:
                reward += 10

            self.replay_memory.append([current_state, bestAction, reward, new_state, done])

            self.train_buffer()

            sum_reward += reward
            current_state = new_state

            if done:
                break

        if done:
            print(f'Episode: {eps} Status: FAIL! with Reward: {sum_reward} & Exploration Rate: {np.around(self.epsilon,4)}')
        else:
            print(f'Episode: {eps} Status: SUCCESS! Episode Time: {self.len_episode} with Reward: {sum_reward} & Exploration Rate: {self.epsilon}')
            self.train_network.save('./train_networkInEPS{}.h5'.format(eps))
        
        # Transfer the weights to 'target_network'
        self.target_network.set_weights(self.train_network.get_weights())

        # Exploration Decay
        self.epsilon -= self.epsilon_decay
        
    
##############################################################################
    def train(self):
        """ Initiate gameplay """
        
        for episode in range(self.n_episodes):
            current_state=env.reset().reshape(1,2)
            self.run(current_state, episode)


##############################################################################
# Main!
##############################################################################
if __name__ == "__main__":
    mountain_car = DQN_Agent(env=gym.make('MountainCar-v0'))
    mountain_car.train()