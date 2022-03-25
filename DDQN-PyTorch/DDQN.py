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
                 density: int = 64,
                 learning_rate: float = 1e-3,
                 name: str = '') -> None:
        super(DuelingDeepQNetwork, self).__init__()

        self.name = name

        self.H1 = torch.nn.Linear(input_dimension, density)
        self.H2 = torch.nn.Linear(density, density)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.H3 = torch.nn.Linear(density, density)
        self.action = torch.nn.Linear(density, action_dimension)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.to(device)

    def forward(self, state) -> torch.Tensor:
        x = F.relu(self.H1(state))
        x = F.relu(self.H2(x))
        x = self.dropout(x)
        x = F.relu(self.H3(x))

        return self.action(x)

    def save_checkpoint(self, path: str = ''):
        torch.save(self.state_dict(), os.path.join(path, self.name + '.pth'))

    def load_checkpoint(self, path: str = ''):
        self.load_state_dict(torch.load(os.path.join(path, self.name + '.pth')))


class Agent():
    def __init__(self,
                 env: Env,
                 n_games: int = 10,
                 batch_size: int = 64,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 eps_min=0.01,
                 eps_dec=1e-3,
                 tau=1e-3):

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

        self.learn_step_counter = 0
        self.indices = np.arange(self.batch_size)

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
        if np.random.random() > self.epsilon:
            self.online_network.eval()

            state = torch.as_tensor(observation, dtype=torch.float32, device=device)
            with torch.no_grad():
                action = self.online_network.forward(state)
                action = torch.argmax(action, dim=-1)
                action = action.cpu().numpy()

        else:
            action = self.env.action_space.sample()

        return action

    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.memory.add(state, action, reward, next_state, done)

    def update_networks(self, tau) -> None:
        for online_weights, target_weights in zip(self.online_network.parameters(), self.target_network.parameters()):
            online_weights.data.copy_(tau * online_weights.data + (1 - tau) * target_weights.data)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self) -> None:
        self.online_network.save_checkpoint()
        self.target_network.save_checkpoint()

    def load_models(self) -> None:
        self.online_network.load_checkpoint()
        self.target_network.load_checkpoint()

    def optimize(self):
        if self.memory.__len__() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.as_tensor(np.vstack(states), dtype=torch.float32, device=device)
        rewards = torch.as_tensor(np.vstack(rewards), dtype=torch.float32, device=device)
        dones = torch.as_tensor(np.vstack(dones), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.vstack(actions), dtype=torch.float32, device=device)
        next_states = torch.as_tensor(np.vstack(next_states), dtype=torch.float32, device=device)

        self.online_network.eval()
        self.target_network.eval()

        predicted_targets = self.online_network(states).gather(1, actions.long())

        with torch.no_grad():
            labels_next = self.target_network(next_states).max(dim=1)[0].unsqueeze(1)

        expected_Q_values = rewards + (1 - dones) * self.gamma * labels_next

        loss = F.huber_loss(predicted_targets, expected_Q_values)

        self.online_network.train()
        self.online_network.optimizer.zero_grad()
        loss.backward()
        self.online_network.optimizer.step()

        self.learn_step_counter += 1

        self.update_networks(tau=self.tau)
        self.decrement_epsilon()
