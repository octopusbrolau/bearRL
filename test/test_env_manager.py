from bear.env_manager.manager import BaseEnvManager, DummyEnvManager, \
    ShmemEnvManager, SubprocEnvManager, RayEnvManager
import gym


if __name__ == "__main__":
    env_num = 4
    envs = SubprocEnvManager([lambda: gym.make('CartPole-v0') for _ in range(env_num)],
                             wait_num=env_num)

    obs = envs.reset()
    obs, reward, done, info = envs.step([1]*env_num)
    envs.render()
    envs.close()