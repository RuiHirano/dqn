from abc import *
from typing import NamedTuple
import torch
from collections import namedtuple
import random
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import os
import copy
from itertools import count
import time
import matplotlib.pyplot as plt
from .replay_memory import Transition, IReplayMemory
from .dqn import IBrain
import ray

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################
#####        Learner         ######
#################################

class ILearner(metaclass=ABCMeta):
    @abstractmethod
    def optimize(self):
        '''Q関数の最適化'''
        pass
    @abstractmethod
    def update_target_model(self):
        '''Target Networkの更新'''
        pass
    @abstractmethod
    def memorize(self, state, action, next_state, reward):
        '''ReplayMemoryへの保存'''
        pass
    @abstractmethod
    def decide_action(self):
        '''行動の決定'''
        pass
    @abstractmethod
    def save_model(self):
        '''modelの保存'''
        pass
    @abstractmethod
    def read_model(self):
        '''modelの保存'''
        pass
    @abstractmethod
    def predict(self):
        '''推論'''
        pass

class LearnerParameter(NamedTuple):
    batch_size: int
    gamma : float
    eps_start : float
    eps_end: float
    eps_decay: int
    replay_memory: IReplayMemory
    net: nn.Module
    multi_step_bootstrap: bool = False
    num_multi_step_bootstrap: int = 5

@ray.remote
class Learner(ILearner):
    def __init__(self, brain: IBrain):
        self.brain = brain

    def update_network(self):
        pass



#################################
#####        Actor         ######
#################################

class IActor(metaclass=ABCMeta):
    @abstractmethod
    def learn(self):
        '''Q関数の更新'''
        pass

@ray.remote
class Actor(IActor):
    def __init__(self, pid, epsilon, gamma, env_name):
        
        self.pid = pid
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.action_space = self.env.action_space.n

        self.q_network = QNetwork(self.action_space)
        self.epsilon = epsilon
        self.gamma = gamma
        self.buffer = []

        self.state = self.env.reset()
        self.define_network()

        self.episode_rewards = 0

    def define_network(self):
        #: ActorからはGPUを見えなくする
        tf.config.set_visible_devices([], 'GPU')
        env = gym.make(self.env_name)
        state = env.reset()
        self.q_network(np.atleast_2d(state))

    def rollout(self, current_weights):
        #: グローバルQ関数と重みを同期
        self.q_network.set_weights(current_weights)

        #: rollout 100step
        for _ in range(100):
            state = self.state
            action = self.q_network.sample_action(state, self.epsilon)
            next_state, reward, done, _ = self.env.step(action)
            self.episode_rewards += reward
            transition = (state, action, reward, next_state, done)
            self.buffer.append(transition)

            if done:
                print(self.episode_rewards)
                self.state = self.env.reset()
                self.episode_rewards = 0
            else:
                self.state = next_state

        #: 初期優先度の計算
        states = np.vstack([transition[0] for transition in self.buffer])
        actions = np.array([transition[1] for trainsition in self.buffer])
        rewards = np.vstack([transition[2] for trainsition in self.buffer])
        next_states = np.vstack([transition[3] for transition in self.buffer])
        dones = np.vstack([transition[4] for transition in self.buffer])

        next_qvalues = self.q_network(next_states)
        next_actions = tf.cast(tf.argmax(next_qvalues, axis=1), tf.int32)
        next_actions_onehot = tf.one_hot(next_actions, self.action_space)
        next_maxQ = tf.reduce_sum(
            next_qvalues * next_actions_onehot, axis=1, keepdims=True)

        TQ = rewards + self.gamma * (1 - dones) * next_maxQ

        qvalues = self.q_network(states)
        actions_onehot = tf.one_hot(actions, self.action_space)
        Q = tf.reduce_sum(qvalues * actions_onehot, axis=1, keepdims=True)

        td_errors = (TQ - Q).numpy().flatten()
        transitions = self.buffer
        self.buffer = []

        return td_errors, transitions, self.pid
        
    
#################################
#####        Trainer       ######
#################################

class ITrainer(metaclass=ABCMeta):
    @abstractmethod
    def learn(self):
        '''Q関数の更新'''
        pass

class Trainer(ITrainer):
    def __init__(self, learner, actor, memory):
        self.learner = learner
        self.actor = actor
        self.memory = memory

    def train(self):
        ray.init()
        start = time.time()
        history = []

        epsilons = np.linspace(0.01, 0.5, num_actors)
        actors = [Actor.remote(pid=i, env_name=env_name, epsilon=epsilons[i], gamma=gamma)
                for i in range(num_actors)]
        
        #: sizeは2の累乗を指定
        replay = Replay(buffer_size=2**14)

        learner = Learner.remote(env_name=env_name, gamma=gamma)
        current_weights = ray.get(learner.define_network.remote())
        current_weights = ray.put(current_weights)

        tester = Tester.remote(env_name=env_name)

        wip_actors = [actor.rollout.remote(current_weights) for actor in actors]
        
        #: まずはある程度遷移情報を蓄積
        for _ in range(30):
            finished, wip_actors = ray.wait(wip_actors, num_returns=1)
            td_errors, transitions, pid = ray.get(finished[0])
            replay.add(td_errors, transitions)
            wip_actors.extend([actors[pid].rollout.remote(current_weights)])
        
        #: Leanerでのネットワーク更新を開始
        minibatchs = [replay.sample_minibatch(batch_size=32) for _ in range(16)]
        wip_learner = learner.update_network.remote(minibatchs)
        minibatchs = [replay.sample_minibatch(batch_size=32) for _ in range(16)]
        wip_tester = tester.test_play.remote(current_weights, epsilon=0.01)

        update_cycles = 1
        actor_cycles = 0
        while update_cycles <= 200:
            actor_cycles += 1
            finished, wip_actors = ray.wait(wip_actors, num_returns=1)
            td_errors, transitions, pid = ray.get(finished[0])
            replay.add(td_errors, transitions)
            wip_actors.extend([actors[pid].rollout.remote(current_weights)])
            
            #: Learnerのタスク完了判定
            finished_learner, _ = ray.wait([wip_learner], timeout=0)
            if finished_learner:
                current_weights, indices, td_errors = ray.get(finished_learner[0])
                wip_learner = learner.update_network.remote(minibatchs)
                current_weights = ray.put(current_weights)
                #: 優先度の更新とminibatchの作成はlearnerよりも十分に速いという前提
                replay.update_priority(indices, td_errors)
                #: つぎのミニバッチセットをあらかじめ用意しておく
                minibatchs = [replay.sample_minibatch(batch_size=32) for _ in range(16)]
                
                print("Actorが遷移をReplayに渡した回数：", actor_cycles)
                update_cycles += 1
                actor_cycles = 0

                if update_cycles % 5 == 0:
                    #:学習状況のtest
                    test_score = ray.get(wip_tester)
                    print(update_cycles, test_score)
                    history.append((update_cycles-5, test_score))
                    wip_tester = tester.test_play.remote(current_weights, epsilon=0.01)

        wallclocktime = round(time.time() - start, 2)
        cycles, scores = zip(*history)
        plt.plot(cycles, scores)
        plt.title(f"total time: {wallclocktime} sec")
        plt.ylabel("test_score(epsilon=0.01)")
        plt.savefig("history.png")