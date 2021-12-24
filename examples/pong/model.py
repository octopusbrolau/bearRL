from bear.model.base import BaseModel
from examples.pong.network import Extractor, Actor, Critic
from typing import Optional, Union
import torch
import os


class ACModel(BaseModel):
    def __init__(self, model_type: str = "ac", **kwargs):
        super(ACModel, self).__init__()
        hidden_shape = kwargs['hidden_size']
        feat_shape = kwargs['feat_size']
        act_shape = kwargs['act_size']
        self.extractor = Extractor(feat_shape, hidden_shape)
        self.actor = Actor(self.extractor, act_shape)
        if model_type == "ac":
            self.critic = Critic(self.extractor)
        self.device = "cpu"

    def set_device(self, device: Union[str, torch.device] = "cpu"):
        self.device = device
        self.extractor.to(device)
        if self.actor is not None:
            self.actor.to(device)
        if self.critic is not None:
            self.critic.to(device)

    def save(self, pt_path):
        torch.save(self.extractor.state_dict(), os.path.join(pt_path, "extractor.pt"))
        torch.save(self.actor.state_dict(), os.path.join(pt_path, "actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(pt_path, "critic.pt"))

    def load(self, pt_path):
        self.extractor.load_state_dict(torch.load(os.path.join(pt_path, 'extractor.pt')))
        self.actor.load_state_dict(torch.load(os.path.join(pt_path, 'actor.pt')))
        self.critic.load_state_dict(torch.load(os.path.join(pt_path, 'critic.pt')))





