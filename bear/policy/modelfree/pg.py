from bear.policy.base import BasePolicy
from typing import Dict, Any, Optional
import numpy as np
import torch
from bear.model.base import BaseModel


class PGPolicy(BasePolicy):
    """
    REINFORCE: MC based PG
    广义优势函数下PG：
    PG+MC:
        REINFORCE
        PPO1 + Advantage
        PPO2 + Advantage: PPO_CLIP
    PG+TD:
        PPO + TD residual(PPO with Actor Critic Style)
    """

    def __init__(self,
                 model: BaseModel,
                 optim: torch.optim.Optimizer,
                 dist_fn: torch.distributions.Distribution,
                 discount_factor: float = 0.99,
                 gae_lambda: float = 1.0,
                 device: str = "cpu",
                 returns_norm: bool = False,
                 **kwargs: Any,
                 ):
        self.device = torch.device(device)
        assert model is not None or model.actor is not None, "model and model.actor must not be None in PGPolicy"
        self.model = model
        # self.model.set_device(self.device)
        self.optim = optim
        self.dist_fn = dist_fn
        assert (
                0.0 <= discount_factor <= 1.0
        ), "discount factor should be in [0, 1]"
        self.gamma = discount_factor
        assert 0.0 <= gae_lambda <= 1.0, "GAE lambda should be in [0, 1]."
        self.gae_lambda = gae_lambda
        self.returns_norm = returns_norm

    def choose_action(self, obs: np.ndarray):
        """ only for discrete actions
            for continuous actions:
                need rewrite this function
        """
        with torch.no_grad():
            logits = self.model.actor(torch.from_numpy(obs).to(self.device).float())
            if isinstance(logits, tuple):
                return logits[0].cpu().numpy()
            else:
                return logits.cpu().numpy()

    def learn(self, sample_feats: np.ndarray, sample_action: np.ndarray, sample_reward: np.ndarray,
              sample_next_feats: np.ndarray, sample_done: np.ndarray, batch_size: int) -> Dict[str, float]:

        returns = self.compute_episodic_return(sample_reward, sample_done,
                                               self.gamma, self.gae_lambda,
                                               returns_norm=self.returns_norm)

        feats, act_target, returns = self.to_torch([sample_feats, sample_action, returns],
                                                   device=self.device)
        act_target = act_target.long()
        act_pred = self.model.actor(feats)
        # loss = (F.cross_entropy(act_pred, act_target, reduce=False) * returns).mean()

        # another way to compute loss
        if isinstance(act_pred, tuple):
            dist = self.dist_fn(*act_pred)
        else:
            dist = self.dist_fn(act_pred)

        log_prob = dist.log_prob(act_target)
        loss = torch.mean(-1 * log_prob * returns)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return {"loss": loss.item()}
