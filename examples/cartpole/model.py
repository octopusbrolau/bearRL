from bear.model.base import BaseModel
from examples.cartpole.network import MLPExtractor, MLPActor, MLPCritic
from typing import Optional, Union
import torch


class ACModel(BaseModel):
    def __init__(self, model_type: str = "ac", **kwargs):
        super(ACModel, self).__init__()
        hidden_shape = kwargs['hidden_size']
        feat_shape = kwargs['feat_size']
        act_shape = kwargs['act_size']
        self.extractor = MLPExtractor(feat_shape, hidden_shape)
        self.actor = MLPActor(self.extractor, hidden_shape, act_shape)
        if model_type == "ac":
            self.critic = MLPCritic(self.extractor, hidden_shape)

    def set_device(self, device: Union[str, torch.device] = "cpu"):
        self.extractor.to(device)
        if self.actor is not None:
            self.actor.to(device)
        if self.critic is not None:
            self.critic.to(device)





