from typing import Union
import numpy as np


class Transition(object):
    def __init__(self):
        self._obs = np.array([1, 2, 3])
        self._act = None
        self._reward = float("inf")
        self._next_obs = None
        self._done = False
        self._value = float("inf")

    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, exp_reward: Union[int, float]):
        self._reward = float(exp_reward)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, exp_value: Union[int, float]):
        self._value = float(exp_value)

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
    def obs(self, exp_obs):
        self._obs = exp_obs

    @property
    def next_obs(self):
        return self._next_obs

    @next_obs.setter
    def next_obs(self, exp_next_obs):
        self._next_obs = exp_next_obs

    @property
    def act(self):
        return self._act

    @act.setter
    def act(self, exp_act):
        self._act = exp_act


class Experience:
    """
    Experience is a sequence of Transitions
    """

    def __init__(self):
        self.trans_list = []
        self.fk = None

    def add(self, item):
        self.trans_list.append(item)
        self.fk = item.obs

    def test(self, trans: Transition ):
        pass

    def reset(self):
        self.trans_list = []

    def __getitem__(self, index):
        return self.trans_list[index]


if __name__ == "__main__":

    buffer = Experience()
    for i in range(1):
        t = Transition()
        t.reward = i
        buffer.add(t)
    buffer.trans_list[0].obs =  np.array([3,4,5])
    print(t.obs)
    print(buffer.fk)
    t.obs = np.array([7,8,9])
    print(buffer.trans_list[0].obs)
    buffer.test(111)
    print('end')