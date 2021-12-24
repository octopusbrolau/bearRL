from bear.policy.modelfree.pg import PGPolicy
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from bear.model.base import BaseModel
# from torch.distributions import Categorical


class A2CPolicy(PGPolicy):
    """
        PG + GAE
        also need collect one n_episode experience
    """

    def __init__(self,
                 model: BaseModel,
                 optim: torch.optim.Optimizer,
                 dist_fn: torch.distributions.Distribution,
                 discount_factor: float = 0.99,
                 gae_lambda: float = 0.95,
                 device: str = "cpu",
                 returns_norm: bool = False,
                 value_loss_weight: float = 0.5,
                 entropy_loss_weight: float = 0.001,
                 **kwargs: Any,
                 ):
        super().__init__(model, optim, dist_fn, discount_factor, gae_lambda,
                         device, returns_norm, **kwargs)
        assert model.critic is not None, \
            "model.critic must not be None in A2CPolicy"
        self.value_loss_weight = value_loss_weight
        self.entropy_loss_weight = entropy_loss_weight

    def learn(self, sample_feats: np.ndarray, sample_action: np.ndarray, sample_reward: np.ndarray,
              sample_next_feats: np.ndarray, sample_done: np.ndarray, batch_size: int) -> Dict[str, float]:

        feats, next_feats, act_target = self.to_torch([sample_feats, sample_next_feats, sample_action], device=self.device)

        with torch.no_grad():
            v_ = self.model.critic(next_feats).cpu().numpy()

            returns = self.compute_episodic_return(sample_reward, sample_done,
                                                   self.gamma, self.gae_lambda,
                                                   returns_norm=False,
                                                   v_s_=v_)
            target_value = self.to_torch([returns])[0].to(self.device)

        act_target = act_target.long()
        act_pred = self.model.actor(feats)

        if isinstance(act_pred, tuple):
            dist = self.dist_fn(*act_pred)
        else:
            dist = self.dist_fn(act_pred)

        act_log_prob = dist.log_prob(act_target)

        pred_value = self.model.critic(feats)
        # advantage = target_value - pred_value
        # without grad broadcast using detach()
        advantage = (target_value - pred_value).detach()
        if self.returns_norm:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

        act_loss = torch.mean(-1 * act_log_prob * advantage)
        value_loss = F.mse_loss(target_value, pred_value)
        entropy_loss = dist.entropy().mean()
        loss = act_loss + self.value_loss_weight * value_loss + self.entropy_loss_weight * entropy_loss
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return {"loss": loss.item(),
                "act_loss": act_loss.item(),
                "value_loss": value_loss.item(),
                "entropy_loss": entropy_loss.item()}
