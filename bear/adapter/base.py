from abc import ABC, abstractmethod
import numpy as np
from typing import Any, List


class BaseAdapter(ABC):
    """
    data pre-processing interface between Env and Policy
    """
    @abstractmethod
    def adapt_obs(self, obs: Any) -> np.ndarray:
        """ preprocess the vectorized reward collected by EnvManager.step()
        in order to be stored in the Transition and then directly used by  policy Networks

        it is the data preprocess function in deep learning

        obs is vectorized obs gotten by env.step(), for example: [obs1, obs2,...]
        obs may be any type (dict, ndarray, tuple ...)

        typical process:
            obs -> ndarray-> normalization-> ...-> feat (must be np.ndarray)

        :param obs: observations gotten by EnvManager.step()
        :return feat: feat can be directly put into policy network
        """
        return obs

    @abstractmethod
    def adapt_reward(self, reward: List[float]) -> np.ndarray:
        """ preprocess the vectorized reward collected by EnvManager.step()
            typical process:
                reward -> normalization(optical) -> rescale->... -> reward (must be np.ndarray)
        """
        return reward

    @abstractmethod
    def adapt_action(self, act: Any) -> np.ndarray:
        """ postprocess the vectorized action executed by EnvManager.step(act)
            in order to be stored in the Transition and then directly used by some policy Networks

        act is vectorized action gotten by env.step(), for example: [act1, act2,...]
        act may be any type (dict, ndarray, tuple ...)

        typical process:
            act -> ndarray-> normalization(or one-hot encoding)-> ...-> action_label (must be np.ndarray)

        """
        return act

    def reset(self):
        pass


