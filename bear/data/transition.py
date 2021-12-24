from typing import Union
import numpy as np
from bear.adapter.base import BaseAdapter


class Transition(object):
    """The minimal internal data structure
    Transition is a collection of <s,a,r,s',done>
    _obs, _act, _next_obs: np.ndarray ( so need a preprocess filter)
    _reward: type float
    _done : type bool
    """

    def __init__(self, ba: BaseAdapter = None):
        self._obs = None
        self._act = None
        self._reward: float = float("inf")
        self._next_obs = None
        self._done: bool = False
        self.adapter = ba

    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, trans_reward: Union[int, float]):
        if self.adapter is None:
            self._reward = float(trans_reward)
        else:
            self._reward = self.adapter.adapt_reward(float(trans_reward))

    @property
    def done(self):
        return self._done

    @done.setter
    def done(self, exp_done: bool):
        self._done = exp_done

    @property
    def obs(self):
        return self._obs

    @obs.setter
    def obs(self, trans_obs: np.ndarray):
        if self.adapter is None:
            self._obs = trans_obs
        else:
            self._obs = self.adapter.adapt_obs(trans_obs)

    @property
    def next_obs(self):
        return self._next_obs

    @next_obs.setter
    def next_obs(self, trans_next_obs: np.ndarray):
        if self.adapter is None:
            self._next_obs = trans_next_obs
        else:
            self._next_obs = self.adapter.adapt_obs(trans_next_obs)

    @property
    def act(self):
        return self._act

    @act.setter
    def act(self, trans_act: np.ndarray):
        if self.adapter is None:
            self._act = trans_act
        else:
            self._act = self.adapter.adapt_action(trans_act)


class AdaptedTransition(Transition):
    def __init__(self, adapter: BaseAdapter):
        # TODO: check if adapter has all needed functions
        super().__init__(adapter)

