
"""
refer: https://blog.csdn.net/cat_ziyan/article/details/101712107
    why choosing Pong-V4?
    why using framestack?
    why using wrap_deepmind?

"""
from examples.common.env.atari_wrappers import wrap_deepmind
import gym
env = gym.make("Pong-v4")
env = wrap_deepmind(env, dim=84, framestack=True, obs_format='NCHW')
obs = env.reset()
env.render()
print(env.action_space.shape)