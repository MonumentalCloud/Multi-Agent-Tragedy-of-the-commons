import gym
from gym import spaces
from scipy.stats import entropy

class VoteEnv(gym.Env):
  def __init__(self, dataset_file_path):

    self.action_space = spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)
    self.observation_space = spaces.Box(low=0, high=255, shape=(256,256,3), dtype=np.uint8)
    self.reward_range = (-200, 200)
    self.max_step = 30
    self.curr_step = 1
    self.dataset = dataset_file_path
    self.batch = self._sample(self.max_step+1)

    self._last = False
    self.reset()

  def reset(self):
    self.curr_step = 1

    #sample new training episode images
    batch_size = self.max_step
    self.batch = self._sample(batch_size)

    return 

  def _sample(self, batch_size):
    batch = []
    with h5py.File(FilePath, 'r') as f:
      max_size = f['input_image'].shape[0]
      sample_idx = random.sample(range(max_size), batch_size)

      for i in sample_idx:
        batch.append(f['input_image'][i])

    return batch

  def _next_observation(self):
    #currently returns random image from the dataset
    obs = torch.FloatTensor(self.batch[self.curr_step-1])
    return torch.movedim(obs, -1,0)

  def _calculate_reward(self, actions):
    #just an inverse of entropy
    return 1/torch.var(actions)


  def step(self, actions):
    self.curr_step += 1

    reward = self._calculate_reward(torch.tensor(actions))
    obs = self._next_observation()
    done = self.curr_step == self.max_step

    return obs, reward, done, {}