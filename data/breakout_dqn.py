import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import gym
import numpy as np
from typing import NamedTuple
from gym import spaces
from gym.spaces.box import Box
from PIL import Image

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#################################
#####      Environment     ######
#################################
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        '''工夫1のNo-Operationです。リセット後適当なステップの間何もしないようにし、
        ゲーム開始の初期状態を様々にすることｆで、特定の開始状態のみで学習するのを防ぐ'''
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(
                1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        '''工夫2のEpisodic Lifeです。1機失敗したときにリセットし、失敗時の状態から次を始める'''
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        '''5機とも失敗したら、本当にリセット'''
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        '''工夫3のMax and Skipです。4フレーム連続で同じ行動を実施し、最後の3、4フレームの最大値をとった画像をobsにする'''
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        '''工夫4のWarp frameです。画像サイズをNatureのDQN論文と同じ84x84の白黒にします'''
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        transform = T.Compose([T.ToPILImage(),T.Grayscale(),
                    T.Resize((self.width, self.height), interpolation=Image.CUBIC),
                    T.ToTensor()])
        observation = torch.from_numpy(frame.transpose(2,0,1).astype(np.float32)).clone()
        observation = transform(observation).unsqueeze(0)
        return observation

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        '''PyTorchのミニバッチのインデックス順に変更するラッパー'''
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        observation = torch.from_numpy(observation.transpose(2, 0, 1).astype(np.float32)).clone().unsqueeze(0)
        return observation



class EnvParameter(NamedTuple):
    max_lot: int    # 最大数量
    spread: int     # スプレッド
    window_size: int
        
class BreakoutEnv(gym.Wrapper):
    def __init__(self):
        env = gym.make('Breakout-v0').unwrapped
        gym.Wrapper.__init__(self, env)
        self.env = NoopResetEnv(self.env, noop_max=30)
        self.env = MaxAndSkipEnv(self.env, skip=4)
        self.env = EpisodicLifeEnv(self.env)
        self.env = WarpFrame(self.env)
        #self.env = WrapPyTorch(self.env)
        
    def reset(self):
        observation = self.env.reset() # (210, 160, 3) (h,w,c)
        #plt.imshow(observation)
        #plt.show()
        state = observation
        return state
        
    def step(self, action): 
        observation, reward, done, info = self.env.step(action)
        state = observation

        if done:
            state = None

        return state, reward, done, info
        
    def get_screen(self, observation):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = observation.transpose((2, 0, 1)) # (3, 210, 160)  (c,h,w)
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]

        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        resize = T.Compose([T.ToPILImage(),T.Grayscale(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
        #img = resize(screen).squeeze(0).numpy()
        #print(img.shape)
        #plt.imshow(img)
        #plt.show()
        # Resize, and add a batch dimension (BCHW)
        return resize(screen).unsqueeze(0).to(device) # (1, 3, 40, 76)  (b,c,h,w)

#################################
#####         Net          ######
#################################
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

def get_env_net():
    env = BreakoutEnv()
    num_actions = env.action_space.n
    init_screen = env.reset()
    _, ch, screen_height, screen_width = init_screen.shape
    net = DQN(screen_height, screen_width, num_actions)
    return env, net