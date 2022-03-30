from typing import Tuple
import os
import numpy as np
from gym import Env

import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.losses import huber_loss

from replay_buffers.Uniform import ReplayBuffer


class DuelingDeepQNetwork(Model):
    def __init__(self,
                 action_dimension: int,
                 density: int = 32,
                 learning_rate: float = 1e-3,
                 name: str = '') -> None:
        super(DuelingDeepQNetwork, self).__init__()

        self.net_name = name
        self.optimizer = Adam(learning_rate=learning_rate)

        self.H1 = Dense(density, activation='relu')
        self.H2 = Dense(density, activation='relu')
        self.H3 = Dense(action_dimension)

    @tf.function()
    def call(self, state):
        state = self.H1(state)
        state = self.H2(state)

        return self.H3(state)


class Agent():
    def __init__(self,
                 env: Env,
                 n_games: int = 10,
                 batch_size: int = 128,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 eps_min: float = 0.01,
                 eps_dec: float = 1e-3,
                 training: bool = True):

        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = learning_rate

        self.action_dim = env.action_space.n
        self.input_dim = env.observation_space.shape[0]
        self.batch_size = batch_size

        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.training = training

        self.memory = ReplayBuffer(self.env._max_episode_steps * n_games)

        self.online_network = DuelingDeepQNetwork(action_dimension=self.action_dim,
                                                  name='OnlinePolicy')

    def choose_action(self, observation) -> int:
        if self.training:
            if np.random.rand(1) > self.epsilon:
                state = tf.convert_to_tensor([observation], dtype=tf.float32)
                Q = self.online_network(state)
                action = tf.argmax(Q, axis=-1)
                action = action[0].numpy()
            else:
                action = self.env.action_space.sample()

            return action

        else:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            Q = self.online_network(state)
            action = tf.argmax(Q, axis=-1)
            action = action[0].numpy()
            return action

    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.memory.add(state, action, reward, next_state, done)

    def epsilon_update(self) -> None:
        '''Decrease epsilon iteratively'''
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec

    def save_models(self, path) -> None:
        self.online_network.save_weights(os.path.join(path, self.online_network.net_name + '.h5'))

    def load_models(self, path) -> None:
        self.online_network.load_weights(os.path.join(path, self.online_network.net_name + '.h5'))

    def optimize(self):
        if self.memory.__len__() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = tf.convert_to_tensor(np.vstack(states), dtype=tf.float32)
        rewards = tf.convert_to_tensor(np.vstack(rewards), dtype=tf.float32)
        dones = tf.convert_to_tensor(np.vstack(dones), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.vstack(actions), dtype=tf.int64)
        next_states = tf.convert_to_tensor(np.vstack(next_states), dtype=tf.float32)

        next_q_values = self.online_network(next_states)
        next_q_values = tf.reduce_max(next_q_values, axis=-1, keepdims=True)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        with tf.GradientTape() as tape:
            current_q_values = self.online_network(states)
            current_q_values = tf.gather_nd(current_q_values, actions, 1)

            loss = huber_loss(tf.squeeze(target_q_values), current_q_values)

        online_network_gradients = tape.gradient(loss, self.online_network.trainable_variables)
        self.online_network.optimizer.apply_gradients(
            zip(online_network_gradients, self.online_network.trainable_variables))

        self.epsilon_update()
