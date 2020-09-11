import gym
import pybullet_envs


import pybullet as p
import numpy as np
import tensorflow as tf

from gym import wrappers
from ac_networks import get_critic, get_actor

import tensorflow.keras.backend as k
from tensorflow.keras.optimizers import Adam
from rl.processors import WhiteningNormalizerProcessor
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess





# Pybullet env
ENV_NAME = 'HalfCheetahBulletEnv-v0'




p.connect(p.DIRECT)
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
actions_n = env.action_space.shape
obs_n = env.observation_space.shape




# Create Actor and Critic networks
k.clear_session()
actor = get_actor(obs_n, actions_n)
critic, action_input = get_critic(obs_n, actions_n)
print(actor.summary())
print(critic.summary())

memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=actions_n, theta=.15, mu=0., sigma=.1)
agent = DDPGAgent(nb_actions=actions_n[0], actor=actor, critic=critic, batch_size=64, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                  random_process=random_process, gamma=.99)

agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=['mse'])

#agent.load_weights('ddpg_' + ENV_NAME + 'weights.h5f')
agent.fit(env, env_name=ENV_NAME, nb_steps=500000, action_repetition=5, visualize=False, verbose=1)



env = wrappers.Monitor(env,'/home/wolfie/PycharmProjects/pythonProject/ddpg_halfcheetah',
                       video_callable=lambda episode_id: True, force=True)


agent.test(env, nb_episodes=5, visualize=False, nb_max_episode_steps=1000, verbose=1)

p.disconnect()