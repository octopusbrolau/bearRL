from bear.policy.modelfree.a2c import A2CPolicy
from typing import Dict, Any, Optional
import numpy as np
import torch
from bear.model.base import BaseModel


class PPO2A2C(A2CPolicy):
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

                 eps_clip: float = 0.2,
                 dual_clip: Optional[float] = None,
                 max_grad_norm: Optional[float] = None,
                 repeat_per_collect: int = 5,
                 value_clip: Optional[float] = None,
                 **kwargs: Any,
                 ):
        super().__init__(model, optim, dist_fn,
                         discount_factor, gae_lambda, device, returns_norm,
                         value_loss_weight, entropy_loss_weight, **kwargs)

        self.eps_clip = eps_clip
        self.repeat = repeat_per_collect
        assert (
                dual_clip is None or dual_clip > 1.0
        ), "Dual-clip PPO parameter should greater than 1.0."
        # dual clip: https://arxiv.org/abs/1912.09729
        self.dual_clip = dual_clip
        self.max_grad_norm = max_grad_norm
        self.value_clip = value_clip

    def learn(self, sample_feats: np.ndarray, sample_action: np.ndarray, sample_reward: np.ndarray,
              sample_next_feats: np.ndarray, sample_done: np.ndarray, batch_size: int) -> Dict[str, float]:

        max_batch_size = batch_size
        sample_length = sample_feats.shape[0]
        sample_old_pred_value, sample_v_, sample_old_act_log_prob = [], [], []
        with torch.no_grad():
            for batch_indices in self.split(sample_length, max_batch_size, shuffle=False):
                _feats, _next_feats, _act_target = self.to_torch([sample_feats[batch_indices],
                                                                  sample_next_feats[batch_indices],
                                                                  sample_action[batch_indices]], device=self.device)

                sample_old_pred_value.append(self.to_numpy(self.model.critic(_feats)))
                sample_v_.append(self.to_numpy(self.model.critic(_next_feats)))
                sample_old_act_pred = self.model.actor(_feats)
                if isinstance(sample_old_act_pred, tuple):
                    old_dist = self.dist_fn(*sample_old_act_pred)
                else:
                    old_dist = self.dist_fn(sample_old_act_pred)

                sample_old_act_log_prob.append(self.to_numpy(old_dist.log_prob(_act_target.long())))

        sample_old_act_log_prob = np.concatenate(sample_old_act_log_prob)
        sample_v_ = np.concatenate(sample_v_)
        sample_old_pred_value = np.concatenate(sample_old_pred_value)

        sample_target_value = self.compute_episodic_return(sample_reward, sample_done,
                                                           self.gamma, self.gae_lambda,
                                                           returns_norm=False,
                                                           v_s_=sample_v_)
        # advantage = target_value - current_pred_value
        # advantage is fixed within one learning phase, so it is also called target_advantage
        # gotten by using fixed old critic model
        sample_advantage = (sample_target_value - sample_old_pred_value)
        if self.returns_norm:
            sample_advantage = (sample_advantage - sample_advantage.mean()) / (sample_advantage.std() + 1e-5)

        stat_losses, stat_act_losses, stat_value_losses, stat_entropy_losses = [], [], [], []

        for i in range(self.repeat):
            for batch_indices in self.split(sample_length, max_batch_size, shuffle=True):
                feats = self.to_torch(sample_feats[batch_indices], device=self.device)
                act_target = self.to_torch(sample_action[batch_indices], device=self.device)
                old_act_log_prob = self.to_torch(sample_old_act_log_prob[batch_indices], device=self.device)
                advantage = self.to_torch(sample_advantage[batch_indices], device=self.device)
                old_pred_value = self.to_torch(sample_old_pred_value[batch_indices], device=self.device)
                target_value = self.to_torch(sample_target_value[batch_indices], device=self.device)

                """compute new act log prob"""
                act_pred = self.model.actor(feats)
                if isinstance(act_pred, tuple):
                    dist = self.dist_fn(act_pred)
                else:
                    dist = self.dist_fn(act_pred)
                act_log_prob = dist.log_prob(act_target.long())
                """compute ppo_clip action loss"""
                ratio = torch.exp(act_log_prob - old_act_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip,
                                    1.0 + self.eps_clip) * advantage
                if self.dual_clip:
                    act_loss = -torch.max(torch.min(surr1, surr2), self.dual_clip * advantage).mean()
                else:
                    act_loss = -torch.min(surr1, surr2).mean()

                """compute value loss and entropy loss"""
                pred_value = self.model.critic(feats)
                if self.value_clip:
                    v_clip = old_pred_value + (pred_value - old_pred_value).clamp(
                        -self.value_clip, self.value_clip)
                    vf1 = (target_value - pred_value).pow(2)
                    vf2 = (target_value - v_clip).pow(2)
                    value_loss = torch.max(vf1, vf2).mean()
                else:
                    value_loss = (target_value - pred_value).pow(2).mean()
                entropy_loss = dist.entropy().mean()
                loss = act_loss + self.value_loss_weight * value_loss + self.entropy_loss_weight * entropy_loss

                self.optim.zero_grad()
                loss.backward()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.actor.parameters()) + list(self.model.critic.parameters()),
                        self.max_grad_norm)
                self.optim.step()

                stat_losses.append(loss.item())
                stat_act_losses.append(act_loss.item())
                stat_value_losses.append(value_loss.item())
                stat_entropy_losses.append(entropy_loss.item())

        return {"loss": np.mean(stat_losses),
                "act_loss": np.mean(stat_act_losses),
                "value_loss": np.mean(stat_value_losses),
                "entropy_loss": np.mean(stat_entropy_losses)}
