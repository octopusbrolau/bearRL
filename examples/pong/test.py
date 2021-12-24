
# https://blog.csdn.net/cat_ziyan/article/details/101712107
from examples.common.env.atari_wrappers import wrap_deepmind
import gym
from bear.env_manager.manager import SubprocEnvManager
# env = gym.make("Pong-v4")
# env = wrap_deepmind(env, dim=84, framestack=True, obs_format='NCHW')

if __name__ == "__main__":
    train_envs_manager = SubprocEnvManager([lambda: wrap_deepmind(gym.make("Pong-v4"), dim=84,
                                                                  framestack=True, obs_format='NCHW')
                                            for _ in range(6)])
    train_envs_manager.seed(1)

    obs = train_envs_manager.reset()

    train_envs_manager.render()
    print(train_envs_manager.action_space.shape)