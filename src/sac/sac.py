class SACAgent:

  def __init__(self, env, gamma, tau, v_lr, q_lr, policy_lr, buffer_maxlen, load_path=False):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.env = env
    self.action_range = [env.action_space.low, env.action_space.high]
    self.obs_dim = env.observation_space.shape
    self.action_dim = env.action_space.shape[0]

    #hyperparameters
    self.gamma = gamma
    self.tau = tau
    self.update_step = 0
    self.delay_step = 2

    #initialize network
    self.value_net = ValueNetwork(self.obs_dim, 1).to(self.device)
    self.target_value_net = ValueNetwork(self.obs_dim, 1).to(self.device)
    self.q_net1 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
    self.q_net2 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
    self.policy_net = GaussianPolicy(self.obs_dim, self.action_dim).to(self.device)

    if load_path:
      checkpoint = torch.load(load_path)
      self.value_net.load_state_dict(checkpoint['value_net'])
      self.targe_value_net.load_state_dict(checkpoint['target_value_net'])
      self.q_net1.load_state_dict(checkpoint['q_net1'])
      self.q_net2.load_state_dict(checkpoint['q_net2'])
      self.policy_net.load_state_dict(checkpoint['policy_net'])

    #copy params to target param
    for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
      target_param.data.copy_(param)

    #initialize optimizers
    self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=v_lr)
    self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=q_lr)
    self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=q_lr)
    self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

    self.replay_buffer = BasicBuffer(buffer_maxlen)

  def get_action(self, state):
    state = torch.FloatTensor(state).to(self.device)
    mean, log_std = self.policy_net.forward(state)
    std = log_std.exp()

    normal = Normal(mean,std)
    z = normal.sample()
    # action = torch.tanh(z)
    action = z
    action = action.cpu().detach().squeeze(0).numpy()

    return action

  def rescale_action(self, action):
    return action * (self.action_range[1] - self.action_range[0]) / 2.0 + (self.action_range[1] + self.action_range[0]) / 2.0

  def update(self, batch_size):
    states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
    states = torch.stack(states).to(self.device)
    actions = torch.FloatTensor(np.array(actions)).to(self.device)
    rewards = torch.FloatTensor(rewards).to(self.device)
    next_states = torch.stack(next_states).to(self.device)
    dones = torch.FloatTensor(dones).to(self.device)
    dones = dones.view(dones.size(0), -1)

    next_actions, next_log_pi = self.policy_net.sample_batch(next_states)
    next_q1 = self.q_net1.sample_batch(next_states, next_actions)
    next_q2 = self.q_net2.sample_batch(next_states, next_actions)
    next_v = self.target_value_net.sample_batch(next_states)

    #value loss
    next_v_target = torch.min(next_q1, next_q2) - next_log_pi
    curr_v = self.value_net.sample_batch(states)
    v_loss = F.mse_loss(curr_v, next_v_target.detach())

    #q loss
    curr_q1 = self.q_net1.sample_batch(states, actions)
    curr_q2 = self.q_net2.sample_batch(states, actions)
    expected_q = rewards + (1-dones) * self.gamma * next_v
    q1_loss = F.mse_loss(curr_q1, expected_q.detach())
    q2_loss = F.mse_loss(curr_q2, expected_q.detach())

    #update value network and q networks
    self.value_optimizer.zero_grad()
    v_loss.backward()
    self.value_optimizer.step()

    self.q1_optimizer.zero_grad()
    q1_loss.backward()
    self.q1_optimizer.step()

    self.q2_optimizer.zero_grad()
    q2_loss.backward()
    self.q2_optimizer.step()

    #delayed update for policy net and target value nets
    if self.update_step % self.delay_step == 0:
      new_actions, log_pi = self.policy_net.sample_batch(states)
      min_q = torch.min(self.q_net1.sample_batch(states, new_actions), self.q_net2.sample_batch(states, new_actions))
      policy_loss = (log_pi - min_q).mean()

      self.policy_optimizer.zero_grad()
      policy_loss.backward()
      self.policy_optimizer.step()

      #target networks
      for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
        target_param.data.copy_(self.tau * param + (1-self.tau) * target_param)
      
    self.update_step += 1
  
  def save_agent(self, path):
    torch.save({
      'value_net': self.value_net,
      'target_value_net': self.target_value_net,
      'q_net1': self.q_net1,
      'q_net2': self.q_net2,
      'policy_net': self.policy_net,
    }, path)



from tqdm import tqdm
from matplotlib import pyplot as plt
def mini_batch_train(env, agents, max_episodes, max_steps=30, batch_size=30):
    episode_rewards = []

    for episode in range(max_episodes):
        # state = env.reset()
        episode_reward = 0
        
        highest_image=False
        lowest_image=False

        lowest_score=False
        highest_score=False
        
        
        for step in tqdm(range(max_steps)):
            state = env._next_observation()
            #get action
            actions = []
            for agent in agents:
              actions.append(agent.get_action(state).item())
            avg = sum(actions) / len(actions)
            
            # print(f"The scores each agents gave was {actions} with the average of {avg}\n")
            next_state, reward, done, _ = env.step(actions)

            #see if the score they gave is low or high and record
            if avg < lowest_score or lowest_score == False:
              lowest_image = state
              lowest_score = avg
            if avg > highest_score or highest_score == False:
              highest_image = state
              highest_score = avg

            #save it to the replay buffer
            for idx, agent in enumerate(agents):
              agent.replay_buffer.push(state, actions[idx], reward, next_state, done)

            #sum of the rewards in each step
            # print(f"reward in the step {step} is {reward}\n")
            episode_reward += reward

            if len(agents[0].replay_buffer) > batch_size:
              for agent in agents:
                agent.update(batch_size)  

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward) + "\n")
                break

            state = next_state
        
        env.reset()

        plt.imshow(highest_image.permute(1,2,0)/255)
        plt.show()
        plt.imshow(lowest_image.permute(1,2,0)/255)
        plt.show()

    return episode_rewards

