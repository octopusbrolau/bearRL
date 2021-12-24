from bear.policy.modelfree.pg import PGPolicy
import numpy as np
import torch
from typing import Any, Dict, Optional
from bear.model.base import BaseModel


class PPO2Policy(PGPolicy):
    def __init__(self,
                 model: BaseModel,
                 optim: torch.optim.Optimizer,
                 dist_fn: torch.distributions.Distribution,
                 discount_factor: float = 0.99,
                 gae_lambda: float = 0.95,
                 device: str = "cpu",
                 returns_norm: bool = False,
                 eps_clip: float = 0.2,
                 dual_clip: Optional[float] = None,
                 max_grad_norm: Optional[float] = None,
                 repeat_per_collect: int = 5,
                 **kwargs: Any,
                 ):
        super().__init__(model, optim, dist_fn, discount_factor, gae_lambda,
                         device, returns_norm, **kwargs)
        self.eps_clip = eps_clip
        self.repeat = repeat_per_collect
        assert (
                dual_clip is None or dual_clip > 1.0
        ), "Dual-clip PPO parameter should greater than 1.0."
        # dual clip: https://arxiv.org/abs/1912.09729
        self.dual_clip = dual_clip
        self.max_grad_norm = max_grad_norm

    def learn(self, batch_feats: np.ndarray, batch_action: np.ndarray, batch_reward: np.ndarray,
              batch_next_feats: np.ndarray, batch_done: np.ndarray) -> Dict[str, float]:
        advantage = self.compute_episodic_return(batch_reward, batch_done,
                                                 self.gamma, self.gae_lambda,
                                                 returns_norm=self.returns_norm)

        feats, act_target, advantage = self.to_torch([batch_feats, batch_action, advantage],
                                                     device=self.device)
        act_target = act_target.long()

        with torch.no_grad():
            old_act_pred = self.model.actor(feats)
            if isinstance(old_act_pred, tuple):
                old_dist = self.dist_fn(*old_act_pred)
            else:
                old_dist = self.dist_fn(old_act_pred)

            old_act_log_prob = old_dist.log_prob(act_target)

        # only ppo_clip loss
        stat_loss = 0.
        for i in range(self.repeat):

            act_pred = self.model.actor(feats)
            if isinstance(act_pred, tuple):
                dist = self.dist_fn(act_pred)
            else:
                dist = self.dist_fn(act_pred)

            act_log_prob = dist.log_prob(act_target)

            ratio = torch.exp(act_log_prob - old_act_log_prob)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip,
                                1.0 + self.eps_clip) * advantage
            if self.dual_clip:
                loss = -torch.max(torch.min(surr1, surr2), self.dual_clip * advantage).mean()
            else:
                loss = -torch.min(surr1, surr2).mean()

            self.optim.zero_grad()
            loss.backward()
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(),
                                               self.max_grad_norm)
            self.optim.step()
            stat_loss += loss.item()

        return {"loss": stat_loss / self.repeat}
