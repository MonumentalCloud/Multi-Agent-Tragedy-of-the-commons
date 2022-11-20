import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftQNetwork(nn.Module):
  def __init__(self, num_inputs, num_actions, hidden_size = 256, init_w=3e-3):
    super(SoftQNetwork, self).__init__()
    self.w = num_inputs[0]
    self.h = num_inputs[1]
    self.in_channel = num_inputs[2]
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.conv1 = nn.Conv2d(self.in_channel, 256, 3)
    self.conv2 = nn.Conv2d(256, 512, 3)
    self.linear3 = nn.Linear(32514049, 1)

    self.linear3.weight.data.uniform_(-init_w, init_w)
    self.linear3.bias.data.uniform_(-init_w, init_w)

  def forward(self, state, action):

    x = F.relu(self.conv1(state))
    x = F.relu(self.conv2(x))
    x = torch.flatten(x)
    x = torch.cat((x, action.unsqueeze(0))).to(self.device)
    x = self.linear3(x)
    return x

  def sample_batch(self, states, actions):
    q_values = []

    for state, action in zip(states, actions):
      q_values.append(self.forward(state, action))

    return torch.tensor(q_values, requires_grad=True).to(self.device)

from torch.distributions import Normal

class GaussianPolicy(nn.Module):
  def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=3e-3, log_std_min=20, log_std_max=2):
    super(GaussianPolicy, self).__init__()

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.in_channel = num_inputs[-1]

    self.log_std_min = log_std_min
    self.log_std_max = log_std_max

    self.conv1 = nn.Conv2d(self.in_channel, 256, 3)
    self.conv2 = nn.Conv2d(256, 512, 3)

    #mean of the gaussian
    self.mean_linear = nn.Linear(32514048, num_actions)
    self.mean_linear.weight.data.uniform_(-init_w, init_w)
    self.mean_linear.bias.data.uniform_(-init_w, init_w)
    
    #network head for log(covariance) of the gaussian distribution
    self.log_std_linear = nn.Linear(32514048, num_actions)
    self.log_std_linear.weight.data.uniform_(-init_w, init_w)
    self.log_std_linear.bias.data.uniform_(-init_w, init_w)

  
  def forward(self, state):
    x = F.relu(self.conv1(state))
    x = F.relu(self.conv2(x))
    x = torch.flatten(x)

    mean = self.mean_linear(x)
    log_std = self.log_std_linear(x)
    log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

    return mean, log_std

  def sample_batch(self, states, epsilon=1e-6):
    actions = []
    log_pis = []

    for image in states:
      mean, log_std = self.forward(image)
      std = log_std.exp()

      normal = Normal(mean, std)
      z = normal.rsample()
      # action = torch.tanh(z)
      action = z

      log_pi = (normal.log_prob(z) - torch.log(1-(torch.tanh(z)).pow(2) + epsilon)).sum()

      actions.append(action)
      log_pis.append(log_pi)
    
    actions = torch.tensor(actions, requires_grad=True).to(self.device)
    log_pis = torch.tensor(log_pis, requires_grad=True).to(self.device)

    return actions, log_pis

class ValueNetwork(nn.Module):
  def __init__(self, input_dim, output_dim, init_w=3e-3):
    super(ValueNetwork, self).__init__()

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.in_channel = input_dim[-1]
    self.conv1 = nn.Conv2d(self.in_channel, 256, 3)
    self.conv2 = nn.Conv2d(256, 512, 3)
    self.fc3 = nn.Linear(32514048, output_dim)


    self.fc3.weight.data.uniform_(-init_w, init_w)
    self.fc3.bias.data.uniform_(-init_w, init_w)

  def forward(self, state):
    x = F.relu(self.conv1(state))
    x = F.relu(self.conv2(x))
    x = torch.flatten(x)
    x = self.fc3(x)

    return x

  def sample_batch(self, states):
    values = []

    for state in states:
      values.append(self.forward(state))

    return torch.tensor(values, requires_grad=True).to(self.device)

import random
import numpy as np
from collections import deque

class BasicBuffer:
  def __init__(self, max_size):
    self.max_size = max_size
    self.buffer = deque(maxlen=max_size)

  def push(self, state, action, reward, next_state, done):
    experience = (state, action, np.array([reward]), next_state, done)
    self.buffer.append(experience)

  def sample(self, batch_size):
    state_batch = []
    action_batch = []
    reward_batch = []
    next_state_batch = []
    done_batch = []

    batch = random.sample(self.buffer, batch_size)

    for experience in batch:
      state, action, reward, next_state, done = experience
      state_batch.append(state)
      action_batch.append(action)
      reward_batch.append(reward)
      next_state_batch.append(next_state)
      done_batch.append(done)

    return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

  def sample_sequence(self, batch_size):
    state_batch = []
    action_batch = []
    reward_batch = []
    next_state_batch = []
    done_batch = []

    min_start = len(self.buffer) - batch_size
    start = np.random.randing(0, min_start)

    for sample in range(start, start + batch_size):
      state, action, reward, next_state, done = self.buffer[sample]
      state_batch.append(state)
      action_batch.append(action)
      reward_batch.append(reward)
      next_state_batch.append(next_state)
      done_batch.append(done)

    return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

  def __len__(self):
    return len(self.buffer)

import torch.optim as optim