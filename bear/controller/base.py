from bear.env_manager.manager import BaseEnvManager, DummyEnvManager
from bear.data.transition import AdaptedTransition
from bear.data.experience import ListExperience, BaseExperienceBuffer
from bear.policy.base import BasePolicy
from bear.adapter.base import BaseAdapter
from bear.exploration.base import BaseExplorer
from typing import Union, Optional, List
import gym
import numpy as np
import warnings


class Controller(object):
    def __init__(self,
                 policy: BasePolicy,
                 adapter: BaseAdapter,
                 explorer: BaseExplorer,
                 env: Union[gym.Env, BaseEnvManager],
                 buffer: BaseExperienceBuffer,
                 ) -> None:
        if not isinstance(env, BaseEnvManager):
            env = DummyEnvManager([lambda: env])
        self.env_manager = env
        self.env_num = len(self.env_manager)
        # environments that are available in step()
        # this means all environments in synchronous simulation
        # but only a subset of environments in asynchronous simulation
        self._ready_env_ids = np.arange(self.env_num)
        self.is_async = self.env_manager.is_async
        self.buffer = buffer
        self.policy = policy
        self.explorer = explorer
        self._cached_buffer = [ListExperience() for _ in range(self.env_num)]
        self.curr_trans = [AdaptedTransition(adapter) for _ in range(self.env_num)]
        self.reset()

    def reset(self):
        self.reset_env_manager()
        self.reset_buffer()

    def reset_env_manager(self):
        self._ready_env_ids = np.arange(self.env_num)
        obs = self.env_manager.reset()
        for i in range(obs.shape[0]):
            self.curr_trans[i].obs = obs[i]

    def reset_cache(self):
        for cb in self._cached_buffer:
            cb.reset()

    def reset_buffer(self):
        self.buffer.reset()
        self.reset_cache()

    def collect(self, n_step: Optional[int] = None, n_episode: Optional[int] = None):
        assert (n_step is not None and n_episode is None and n_step > 0) or (
                n_step is None and n_episode is not None and n_episode > 0
        ), "Only one of n_step or n_episode is allowed in Controller.collect, "
        f"got n_step = {n_step}, n_episode = {n_episode}."

        self.reset_cache()
        if n_step is not None:
            self.collect_n_step_experience(n_step)
        elif n_episode is not None:
            self.collect_n_episode_experience(n_episode)
        else:
            warnings.warn(
                "Either n_step or n_episode is not None",
                Warning)

    def collect_n_episode_experience(self, n_episode):
        for i in range(n_episode):
            self.reset_env_manager()
            self.collect_one_episode_experience()

    def collect_one_episode_experience(self):
        complete_env_ids = []
        working_env_ids = [idx for idx in self._ready_env_ids if idx not in complete_env_ids]
        while len(working_env_ids) > 0:
            ended_env_ids = self.collect_one_step_experience(working_env_ids)
            complete_env_ids.extend(ended_env_ids)
            working_env_ids = [idx for idx in self._ready_env_ids if idx not in complete_env_ids]

        for env_id in complete_env_ids:
            self.buffer.add(self._cached_buffer[env_id])
        self.reset_cache()

    def collect_n_step_experience(self, n_step):
        complete_env_ids = []
        working_env_ids = [idx for idx in self._ready_env_ids if idx not in complete_env_ids]
        while n_step > 0:
            ended_env_ids = self.collect_one_step_experience(working_env_ids)
            complete_env_ids.extend(ended_env_ids)
            n_step -= 1
        complete_env_ids.extend(self._ready_env_ids)

        for env_id in complete_env_ids:
            self.buffer.add(self._cached_buffer[env_id])
        self.reset_cache()

    def collect_one_step_experience(self, working_env_ids: List[int]) -> List[int]:
        """
        run one step for env in working_env_ids
        update curr_trans and store it into cached_buffer
        return env ids that reach the end of an episode and need to be reset
        before return, reset the ended env ids and update curr_trans
        :param working_env_ids:
        :return: ended_env_ids
        """
        ended_env_ids = []
        obs = [self.curr_trans[i].obs for i in working_env_ids]
        act_logits = self.policy.choose_action(np.asarray(obs))
        act = self.explorer.explore(act_logits)
        next_obs, reward, done, info = self.env_manager.step(act.copy(), id=working_env_ids)
        self._ready_env_ids = np.array([i['env_id'] for i in info])
        for i, env_idx in enumerate(self._ready_env_ids):
            self.curr_trans[env_idx].next_obs = next_obs[i]
            self.curr_trans[env_idx].act = act[i]
            self.curr_trans[env_idx].reward = reward[i]
            self.curr_trans[env_idx].done = done[i]
            self._cached_buffer[env_idx].add(self.curr_trans[env_idx])
            if done[i]:
                # done[i] == True
                ended_env_ids.append(env_idx)
                complete_obs = self.env_manager.reset(env_idx)
                self.curr_trans[env_idx].obs = complete_obs[0].copy()
            else:
                self.curr_trans[env_idx].obs = next_obs[i].copy()

        return ended_env_ids

    def learn(self, batch_size: int = 256):
        sample_feats, sample_action, sample_reward, sample_next_feats, sample_done = self.buffer.sample(batch_size)
        losses = self.policy.learn(sample_feats, sample_action, sample_reward, sample_next_feats, sample_done, batch_size)
        return losses
