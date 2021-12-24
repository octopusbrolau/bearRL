import torch
import torch.nn.functional as F


class MLPExtractor(torch.nn.Module):
    def __init__(self, feat_shape: int, hidden_shape: int):
        super(MLPExtractor, self).__init__()
        self.fc = torch.nn.Linear(feat_shape, hidden_shape)

    def forward(self, feat):
        x = self.fc(feat)
        x = F.relu(x)
        return x


class MLPActor(torch.nn.Module):
    def __init__(self, extractor: MLPExtractor, hidden_shape: int, act_shape: int):
        super(MLPActor, self).__init__()

        self.extractor = extractor
        self.fc_out = torch.nn.Linear(hidden_shape, act_shape)

    def forward(self, feat):
        x = self.extractor(feat)
        out = self.fc_out(x)
        return F.softmax(out, dim=1)


class MLPCritic(torch.nn.Module):
    def __init__(self, extractor: MLPExtractor, hidden_shape: int, last_size: int = 1):
        super(MLPCritic, self).__init__()

        self.extractor = extractor

        self.fc_out = torch.nn.Linear(hidden_shape, last_size)

    def forward(self, feat):
        x = self.extractor(feat)
        out = self.fc_out(x)
        return out.squeeze(dim=-1)
