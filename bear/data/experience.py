from bear.data.transition import Transition
from abc import ABC, abstractmethod
import numpy as np


class ListExperience(object):
    """
    Experience is a sequence of Transitions
    ListExperience stores transitions in origin python list, normally used in EnvManager
    when in MC or TD(lambda>1) mode:
        * it ***must*** stores ordered transitions of an episode from one agent;
        * check it before store in this ListExperience
        * for multi agents, ListExperienceManager should be used as local cache for EnvManager
    when in TD(1) mode:
        it can store transitions of multi agents in one step
    """

    def __init__(self):
        self.trans_list = []

    def add(self, trans: Transition):
        trans_cp = Transition()
        # deep copy np.ndarray
        trans_cp.obs = trans.obs.copy()
        trans_cp.act = trans.act.copy()
        trans_cp.next_obs = trans.next_obs.copy()
        trans_cp.reward = trans.reward
        trans_cp.done = trans.done
        self.trans_list.append(trans_cp)

    def reset(self):
        self.trans_list = []

    def __getitem__(self, index):
        return self.trans_list[index]

    @property
    def size(self):
        return len(self.trans_list)


class BaseExperienceBuffer(ABC):
    def __init__(self, max_size):
        self.max_size = max_size
        self.obs = None
        self.action = None
        self.reward = None
        self.next_obs = None
        self.done = None

        self._curr_size = 0
        self._curr_pos = 0

    def _infer_storage_shape(self, shape: tuple):
        new_shape = [self.max_size]
        new_shape.extend(list(shape))
        return new_shape

    def _init_storage(self, obs_shape: tuple, act_shape: tuple, next_obs_shape: tuple):
        self.obs = np.zeros(self._infer_storage_shape(obs_shape), dtype='float32')
        self.action = np.zeros(self._infer_storage_shape(act_shape), dtype='float32')
        self.next_obs = np.zeros(self._infer_storage_shape(next_obs_shape), dtype='float32')
        self.reward = np.zeros((self.max_size,), dtype='float32')
        self.done = np.zeros((self.max_size,), dtype='bool')

    def _add(self, obs: np.ndarray, act: np.ndarray, reward: float, next_obs: np.ndarray, done: bool):
        if self.obs is None or self.action is None or self.reward is None or self.next_obs is None or self.done is None:
            self._init_storage(obs.shape, act.shape, next_obs.shape)

        self.obs[self._curr_pos] = obs.copy()
        self.action[self._curr_pos] = act.copy()
        self.next_obs[self._curr_pos] = next_obs.copy()
        self.reward[self._curr_pos] = reward
        self.done[self._curr_pos] = done

    def increase_size(self):
        if self._curr_size < self.max_size:
            self._curr_size += 1
        self._curr_pos = (self._curr_pos + 1) % self.max_size

    def add(self, list_exp: ListExperience):
        assert isinstance(list_exp, ListExperience)
        for trans in list_exp:
            self._add(trans.obs, trans.act, trans.reward, trans.next_obs, trans.done)
            self.increase_size()

    def reset(self):
        self._curr_pos = 0
        self._curr_size = 0

    @property
    def size(self):
        return self._curr_size

    @property
    def capacity(self):
        return self.max_size

    @abstractmethod
    def sample(self, batch_size):
        pass


class ReplayExperienceBuffer(BaseExperienceBuffer):
    """
    ReplayExperienceBuffer is data switcher of env and policy
    recv listExperience or List[listExperience] collected by controller, generated by EnvManager
    restore it in a more efficient way for sampling
    be able to sample a batch of data for policy learning
    """
    def __init__(self, max_size):
        super().__init__(max_size)

    def sample(self, batch_size):
        # sample at most batch_size
        assert self._curr_size > 0, "no data can be sampled"
        batch_idx = np.random.randint(self._curr_size, size=batch_size)
        batch_obs = self.obs[batch_idx]
        batch_action = self.action[batch_idx]
        batch_reward = self.reward[batch_idx]
        batch_next_obs = self.next_obs[batch_idx]
        batch_done = self.done[batch_idx]
        return batch_obs, batch_action, batch_reward, batch_next_obs, batch_done


class RolloutExperienceBuffer(BaseExperienceBuffer):
    def __init__(self, max_size):
        super().__init__(max_size)

    def sample(self, batch_size):
        # get all data in a queue way
        assert self._curr_size > 0, "no data can be sampled"
        batch_obs = self.obs[:self._curr_pos]
        batch_action = self.action[:self._curr_pos]
        batch_reward = self.reward[:self._curr_pos]
        batch_next_obs = self.next_obs[:self._curr_pos]
        batch_done = self.done[:self._curr_pos]
        # lazy clear the buffer
        self.reset()
        return batch_obs, batch_action, batch_reward, batch_next_obs, batch_done

    def increase_size(self):
        assert self._curr_size < self.max_size, " num of Experience collected exceeds RolloutExperienceBuffer's capacity"
        self._curr_size += 1
        self._curr_pos = (self._curr_pos + 1) % self.max_size


