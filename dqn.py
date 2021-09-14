import time
import sys
sys.path.append('./../')
from lib.dqn import Trainer, Examiner, Brain, Agent, BrainParameter, TrainParameter
from lib.replay_memory import ReplayMemory, PrioritizedReplayMemory
from lib.util import Color
color = Color()
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import gym
import argparse
from importlib import import_module
import json
import datetime
import os
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file',type=str, default='', help='Config file')
    args = parser.parse_args()

    if args.file == "":
        color.red("Config is not found: Please use -f option")
        sys.exit(1)

    def load_yaml(filename: str):
        with open("{}".format(Path(filename).resolve()), 'r') as f:
            d = yaml.safe_load(f)
        return d
    config = load_yaml(args.file)
    color.green("config: {}".format(json.dumps(config, indent=2)))
    time.sleep(2)

    ''' Memory生成 '''
    memory = ReplayMemory(CAPACITY=config["replay"]["capacity"])
    if config["replay"]["type"] == "PrioritizedExperienceReplay":
        memory = PrioritizedReplayMemory(CAPACITY=config["replay"]["capacity"])

    ''' 環境生成 '''
    module = import_module("data.{}".format(config["info"]["module_name"]))
    env, net = module.get_env_net()
    
    ''' エージェント生成 '''
    brain_param = BrainParameter(
        replay_memory=memory, 
        net=net, 
        batch_size=config["train"]["batch_size"],
        gamma=config["train"]["gamma"],
        eps_start=config["train"]["eps_start"],
        eps_end=config["train"]["eps_end"],
        eps_decay=config["train"]["eps_decay"],
        multi_step_bootstrap=config["train"]["multi_step_bootstrap"],
        num_multi_step_bootstrap=config["train"]["num_multi_step_bootstrap"],
    )
    brain = Brain(brain_param, env.action_space.n)
    agent = Agent(brain)

    name = config["info"]["name"]
    id = name+datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    ''' configの保存 '''
    fn = "./results/{}/config.yaml".format(id)
    dirname = os.path.dirname(fn)
    if os.path.exists(dirname) == False:
        os.makedirs(dirname)
    with open(fn, "w") as yf:
        yaml.dump(config, yf, default_flow_style=False)

    train_mode = config["train"]["train_mode"]
    if train_mode:
        ''' Trainer '''
        trainer = Trainer(id, env, agent)
        train_param = TrainParameter(
            target_update_iter=config["train"]["target_update_iter"],
            num_episode =config["train"]["num_episode"],
            save_iter=config["train"]["save_iter"],
            save_filename=config["train"]["save_filename"],
            render=config["train"]["render"],
        )
        trainer.train(train_param)
    else:
        agent.remember(name="./results/cartpole/20210912_dualingdqn2/cartpole_6000.pth")
        ''' Eval '''
        examiner = Examiner(env, agent)
        examiner.evaluate(config["eval"]["num_episode"], render=config["eval"]["render"])