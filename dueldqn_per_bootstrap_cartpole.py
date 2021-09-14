import sys
sys.path.append('./../')
from lib.dqn import Trainer, Examiner, Brain, Agent, BrainParameter
from lib.replay_memory import PrioritizedReplayMemory

import torch
import torch.nn as nn
import gym

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################
#####      Environment     ######
#################################

class Environment(gym.Wrapper):
    def __init__(self):
        env = gym.make('CartPole-v0').unwrapped
        gym.Wrapper.__init__(self, env)
        self.episode_step = 0
        self.complete_episodes = 0
        
    def step(self, action): 
        observation, reward, done, info = self.env.step(action)
        self.episode_step += 1

        state = torch.from_numpy(observation).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
        state = torch.unsqueeze(state, 0)

        if self.episode_step == 200: # 200以上でdoneにする
            done = True

        if done:
            state = None
            if self.episode_step > 195:
                reward = 1
                self.complete_episodes += 1  # 連続記録を更新
                if self.complete_episodes >= 10:
                    print("{}回連続成功".format(self.complete_episodes))
            else:
                # こけたら-1を与える
                reward = -1
                self.complete_episodes = 0
            
            self.episode_step = 0

        return state, reward, done, info

    def reset(self):
        observation = self.env.reset()
        state = torch.from_numpy(observation).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
        state = torch.unsqueeze(state, 0)
        return state

#################################
#####         Net          ######
#################################

class DuelingDQN(nn.Module):
    '''線形入力でDualingNetworkを搭載したDQN'''
    def __init__(self, num_states, num_actions):
        super(DuelingDQN, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.num_states, 32)
        self.relu = nn.ReLU()
        self.fcV1 = nn.Linear(32, 32)
        self.fcA1 = nn.Linear(32, 32)
        self.fcV2 = nn.Linear(32, 1)
        self.fcA2 = nn.Linear(32, self.num_actions)

    def forward(self, x):
        x = self.relu(self.fc1(x))

        V = self.fcV2(self.fcV1(x))
        A = self.fcA2(self.fcA1(x))

        averageA = A.mean(1).unsqueeze(1)
        return V.expand(-1, self.num_actions) + (A - averageA.expand(-1, self.num_actions))

class DQN(nn.Module):

    def __init__(self, num_states, num_actions):
        super(DQN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_states, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
            #nn.ReLU()
        )

    def forward(self, x):
        x = x.to(device)
        logits = self.linear_relu_stack(x)
        return logits

#################################
#####         Main         ######
#################################

if __name__ == "__main__":
    ''' 環境生成 '''
    env = Environment()
    num_actions = env.action_space.n
    num_states = env.observation_space.shape[0]
    
    train_mode = False
    memory = PrioritizedReplayMemory(CAPACITY=10000)
    net = DuelingDQN(num_states, num_actions)
    ''' エージェント生成 '''
    brain_param = BrainParameter(replay_memory=memory, net=net, batch_size=32, gamma=0.97, eps_start=0.9, eps_end=0.05, eps_decay=200, multi_step_bootstrap=True, num_multi_step_bootstrap=5)
    brain = Brain(brain_param, num_states, num_actions)
    agent = Agent(brain)
    if train_mode:
        ''' Trainer '''
        trainer = Trainer(env, agent)
        trainer.train(3000, save_dir="./results/cartpole/20210912_per_bootstrap_dualingdqn", file_name="cartpole", save_iter=1000, render=False)
    else:
        agent.remember(name="./results/cartpole/20210912_per_bootstrap_dualingdqn/cartpole_3000.pth")
        ''' Eval '''
        examiner = Examiner(env, agent)
        examiner.evaluate(100, render=True)