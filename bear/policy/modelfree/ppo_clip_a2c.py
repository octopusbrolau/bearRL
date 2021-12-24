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

    def learn(self, batch_feats: np.ndarray, batch_action: np.ndarray, batch_reward: np.ndarray,
              batch_next_feats: np.ndarray, batch_done: np.ndarray) -> Dict[str, float]:

        feats, next_feats, act_target = self.to_torch([batch_feats, batch_next_feats, batch_action], device=self.device)
        act_target = act_target.long()
        print("episode length: ", act_target.shape)
        """compute advantage"""
        with torch.no_grad():
            old_pred_value = self.model.critic(feats)
            v_ = self.model.critic(next_feats).cpu().numpy()

            returns = self.compute_episodic_return(batch_reward, batch_done,
                                                   self.gamma, self.gae_lambda,
                                                   returns_norm=False,
                                                   v_s_=v_)
            target_value = self.to_torch([returns])[0].to(self.device)

        # advantage = target_value - current_pred_value
        # advantage is fixed within one learning phase, so it is also called target_advantage
        # gotten by using fixed old critic model
        advantage = (target_value - old_pred_value)
        if self.returns_norm:
            advantage = (advantage - advantage.mean())/(advantage.std() + 1e-5)
        """compute old act log prob"""
        with torch.no_grad():
            old_act_pred = self.model.actor(feats)
            if isinstance(old_act_pred, tuple):
                old_dist = self.dist_fn(*old_act_pred)
            else:
                old_dist = self.dist_fn(old_act_pred)

            old_act_log_prob = old_dist.log_prob(act_target)

        stat_losses, stat_act_losses, stat_value_losses, stat_entropy_losses = [], [], [], []

        for i in range(self.repeat):
            """compute new act log prob"""
            act_pred = self.model.actor(feats)
            if isinstance(act_pred, tuple):
                dist = self.dist_fn(act_pred)
            else:
                dist = self.dist_fn(act_pred)
            act_log_prob = dist.log_prob(act_target)
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
                    list(self.model.actor.parameters())+list(self.model.critic.parameters()),
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
