import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Union
import torch
from numba import njit


class BasePolicy(ABC):
    @abstractmethod
    def choose_action(self, obs: np.ndarray):
        pass

    @abstractmethod
    def learn(self, sample_feats: np.ndarray, sample_action: np.ndarray, sample_reward: np.ndarray,
              sample_next_feats: np.ndarray, sample_done: np.ndarray, batch_size: int) -> Dict[str, float]:
        pass

    @staticmethod
    # @njit
    def compute_episodic_return(rew: np.ndarray,
                                done: np.ndarray,
                                gamma: float,
                                gae_lambda: float,
                                returns_norm: bool = False,
                                v_s_: np.ndarray = None) -> np.ndarray:
        v_s_ = np.zeros_like(rew) if v_s_ is None else v_s_.flatten()
        returns = np.roll(v_s_, 1)
        m = (1.0 - done) * gamma
        delta = rew + v_s_ * m - returns
        m *= gae_lambda
        gae = 0.0
        for i in range(len(rew) - 1, -1, -1):
            gae = delta[i] + m[i] * gae
            returns[i] += gae
        if returns_norm:
            returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())

        return returns.copy()

    @staticmethod
    def to_torch(arr_list: Union[List[np.ndarray], np.ndarray], device: str = "cpu"):
        if isinstance(arr_list, np.ndarray):
            return torch.from_numpy(arr_list).to(device)
        else:
            return [torch.from_numpy(arr).to(device) for arr in arr_list]

    @staticmethod
    def to_numpy(tensor):
        return tensor.cpu().numpy()

    @staticmethod
    def split(length: int, size: int, shuffle: bool = True, merge_last: bool = False):
        """Split whole data into multiple small batches.

        :param length: length of the collected data
        :param int size: divide the data batch with the given size, but one
            batch if the length of the batch is smaller than "size".
        :param bool shuffle: randomly shuffle the entire data batch if it is
            True, otherwise remain in the same. Default to True.
        :param bool merge_last: merge the last batch into the previous one.
            Default to False.
        """

        assert 1 <= size  # size can be greater than length, return whole batch
        if shuffle:
            indices = np.random.permutation(length)
        else:
            indices = np.arange(length)
        merge_last = merge_last and length % size > 0
        for idx in range(0, length, size):
            if merge_last and idx + size + size >= length:
                yield indices[idx:]
                break
            yield indices[idx:idx + size]
