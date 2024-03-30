import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import deque
from utils import DQNbn, DQN, make_env, ReplayMemory, train
import warnings

warnings.simplefilter("ignore", UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.02
EPS_DECAY = 100000
TARGET_UPDATE = 500
RENDER = False
lr = 1e-4
INITIAL_MEMORY = 1_000
MEMORY_SIZE = 10 * INITIAL_MEMORY

policy_net = DQNbn(n_actions=4).to(device)  # убираем действия с FIRE
target_net = DQNbn(n_actions=4).to(device)
# policy_net = DQN(n_actions=4, in_channels=4).to(device)  # убираем действия с FIRE
# target_net = DQN(n_actions=4, in_channels=4).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

steps_done = 0

# create environment
env = gym.make("BreakoutNoFrameskip-v4")
env = make_env(env)

memory = ReplayMemory(MEMORY_SIZE)

train(
    model_name="breakout_dqn",
    env=env,
    n_episodes=10_000,
    memory=memory,
    device=device,
    initial_memory=INITIAL_MEMORY,
    policy_net=policy_net,
    target_net=target_net,
    gamma=GAMMA,
    optimizer=optimizer,
    batch_size=BATCH_SIZE,
    target_update=TARGET_UPDATE,
    eps_end=EPS_END,
    eps_start=EPS_START,
    eps_decay=EPS_DECAY,
    render=False,
)

torch.save(policy_net, "breakout_dqn_model")
