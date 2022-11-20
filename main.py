import h5py
from src.env import VoteEnv
from src.sac.sac import SACAgent
from src.mini_batch_train import mini_batch_train

FilePath = "/content/drive/MyDrive/Marvin/Thesis/dataset/fashiongen_256_256_train.h5"

env = VoteEnv(FilePath)

#SAC 2018 Params
tau = 0.005
gamma = 0.99
value_lr = 3e-3
q_lr = 3e-3
policy_lr = 3e-3
buffer_maxlen = 1000000

agent1 = SACAgent(env, gamma, tau, value_lr, q_lr, policy_lr, buffer_maxlen)
agent2 = SACAgent(env, gamma, tau, value_lr, q_lr, policy_lr, buffer_maxlen)
agent3 = SACAgent(env, gamma, tau, value_lr, q_lr, policy_lr, buffer_maxlen)


agents = [agent1, agent2, agent3]

episode_rewards = mini_batch_train(env, agents, max_episodes=50, max_steps=30, batch_size=30)