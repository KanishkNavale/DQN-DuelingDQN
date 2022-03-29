from typing import Tuple
import os
import numpy as np
from gym import Env

import torch
import torch.nn.functional as F
import torch.optim as optim

from replay_buffers.Uniform import ReplayBuffer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DuelingDeepQNetwork(torch.nn.Module):
    def __init__(self,
                 input_dimension: int,
                 action_dimension: int,
                 density: int = 512,
                 learning_rate: float = 1e-3,
                 name: str = '') -> None:
        super(DuelingDeepQNetwork, self).__init__()

        self.name = name

        self.H1 = torch.nn.Linear(input_dimension, density)
        self.H2 = torch.nn.Linear(density, density)

        self.V1 = torch.nn.Linear(density, density)
        self.V2 = torch.nn.Linear(density, density)

        self.A1 = torch.nn.Linear(density, density)
        self.A2 = torch.nn.Linear(density, density)

        self.value = torch.nn.Linear(density, 1)
        self.advantage = torch.nn.Linear(density, action_dimension)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.to(device)

    def forward(self, state) -> torch.Tensor:

        state = F.relu(self.H1(state))
        state = F.relu(self.H2(state))

        value = F.relu(self.V1(state))
        value = F.relu(self.V2(value))
        value = self.value(value)

        adv = F.relu(self.A1(state))
        adv = F.relu(self.A2(adv))
        adv = self.advantage(adv)

        return value + adv - torch.mean(adv, dim=-1, keepdim=True)

    def pick_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action = torch.argmax(Q, dim=-1)
            return action.cpu().numpy()

    def save_checkpoint(self, path: str = ''):
        torch.save(self.state_dict(), os.path.join(path, self.name + '.pth'))

    def load_checkpoint(self, path: str = ''):
        self.load_state_dict(torch.load(os.path.join(path, self.name + '.pth')))


class Agent():
    def __init__(self,
                 env: Env,
                 n_games: int = 10,
                 batch_size: int = 32,
                 learning_rate: float = 0.5e-4,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 eps_min: float = 0.01,
                 eps_dec: float = 1e-3,
                 tau: float = 0.001,
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
        self.tau = tau

        self.training = training

        self.memory = ReplayBuffer(self.env._max_episode_steps * n_games)

        self.online_network = DuelingDeepQNetwork(input_dimension=self.input_dim,
                                                  action_dimension=self.action_dim,
                                                  learning_rate=learning_rate,
                                                  name='OnlinePolicy')

        self.target_network = DuelingDeepQNetwork(input_dimension=self.input_dim,
                                                  action_dimension=self.action_dim,
                                                  learning_rate=learning_rate,
                                                  name='TargetPolicy')

        self.update_networks(tau=1.0)

    def choose_action(self, observation) -> int:
        if self.training:
            if np.random.rand(1) > self.epsilon:
                self.online_network.eval()
                state = torch.as_tensor(observation, dtype=torch.float32, device=device)
                with torch.no_grad():
                    action = self.online_network.pick_action(state)
            else:
                action = self.env.action_space.sample()

            return action
            
        else:
            state = torch.as_tensor(observation, dtype=torch.float32, device=device)
            with torch.no_grad():
                return self.online_network.pick_action(state)

    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.memory.add(state, action, reward, next_state, done)

    def update_networks(self, tau) -> None:
        for online_weights, target_weights in zip(self.online_network.parameters(), self.target_network.parameters()):
            target_weights.data.copy_(tau * online_weights.data + (1 - tau) * target_weights.data)

    def epsilon_update(self) -> None:
        '''Decrease epsilon iteratively'''
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec

    def save_models(self, path) -> None:
        self.online_network.save_checkpoint(path)
        self.target_network.save_checkpoint(path)

    def load_models(self, path) -> None:
        self.online_network.load_checkpoint(path)
        self.target_network.load_checkpoint(path)

    def optimize(self):
        if self.memory.__len__() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.as_tensor(np.vstack(states), dtype=torch.float32, device=device)
        rewards = torch.as_tensor(np.vstack(rewards), dtype=torch.float32, device=device)
        dones = torch.as_tensor(np.vstack(dones), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.vstack(actions), dtype=torch.float32, device=device)
        next_states = torch.as_tensor(np.vstack(next_states), dtype=torch.float32, device=device)

        self.online_network.train()
        self.target_network.train()

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_values, _ = next_q_values.max(dim=1)
            next_q_values = next_q_values.reshape(-1, 1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        current_q_values = self.online_network(states)
        current_q_values = torch.gather(current_q_values, dim=1, index=actions.long())

        # Compute Huber loss (less sensitive to outliers)
        loss = F.huber_loss(current_q_values, target_q_values)

        self.online_network.optimizer.zero_grad()
        loss.backward()
        self.online_network.optimizer.step()

        self.update_networks(tau=self.tau)
        self.epsilon_update()
