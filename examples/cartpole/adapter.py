from typing import Any, List
import numpy as np
from bear.adapter.base import BaseAdapter


class CartPoleAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()

    def adapt_action(self, act: Any) -> np.ndarray:
        return act

    def adapt_reward(self, reward: List[float]) -> np.ndarray:
        return reward

    def adapt_obs(self, obs: Any) -> np.ndarray:
        return obs



# class CartPoleAdapter(BaseAdapter):
#     def __init__(self):
#         super().__init__()
#         self.his_n = 0
#         self.his_mean = None
#         self.his_std = None
#         self.his_cache = []
#         self.his_cache_n = 10000
#         self.his_warmup_n = 10000
#         self.his_max_n = self.his_cache_n * 10
#         self.his_update_period_n = self.his_cache_n * 1
#         self.his_period_count_n = 0
#
#     def add_obs_to_history(self, obs):
#         if self.his_n >= self.his_max_n:
#             return
#         self.his_cache.append(obs.copy())
#
#         if self.his_n <= 0:
#             if len(self.his_cache) == self.his_warmup_n:
#                 self.his_n = self.his_warmup_n
#                 self.his_mean = np.mean(self.his_cache, axis=0)
#                 self.his_std = np.std(self.his_cache, axis=0, ddof=1)
#                 self.his_cache = []
#                 print("warm up")
#             else:
#                 return
#         else:
#             self.his_period_count_n += 1
#             if len(self.his_cache) == self.his_cache_n:
#                 if self.his_period_count_n < self.his_update_period_n:
#                     self.his_cache = []
#                     return
#                 cache_mean = np.mean(self.his_cache, axis=0)
#                 cache_std = np.std(self.his_cache, axis=0, ddof=1)
#                 new_mean = (self.his_n*self.his_mean + self.his_cache_n*cache_mean)/(self.his_n + self.his_cache_n)
#                 new_sigma2 = (
#                         self.his_n*(self.his_std**2 + (new_mean - self.his_mean)**2) +
#                         self.his_cache_n*(cache_std**2 + (new_mean - cache_mean)**2)
#                 ) / (self.his_n + self.his_cache_n)
#
#                 self.his_n += self.his_cache_n
#                 self.his_mean = new_mean
#                 self.his_std = np.sqrt(new_sigma2)
#                 self.his_cache = []
#                 self.his_period_count_n = 0
#                 print("##########obs stats updated############")
#                 print("count: ", self.his_n)
#                 print("mean: ", self.his_mean)
#                 print("std: ", self.his_std)
#
#             else:
#                 return
#
#     def get_stats(self):
#         return self.his_mean, self.his_std
#
#     def adapt_action(self, act: Any) -> np.ndarray:
#         return act
#
#     def adapt_reward(self, reward: List[float]) -> np.ndarray:
#         return reward
#
#     def adapt_obs(self, obs: Any) -> np.ndarray:
#         self.add_obs_to_history(obs)
#         mean, std = self.get_stats()
#         if mean is not None and std is not None:
#             return (obs - mean) / std
#         else:
#             return obs


