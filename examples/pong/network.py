import torch
import torch.nn.functional as F


class Extractor(torch.nn.Module):
    def __init__(self, feat_shape: int, hidden_shape: int):
        super(Extractor, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=feat_shape, out_channels=hidden_shape, kernel_size=8, stride=4, padding=2)
        self.conv2 = torch.nn.Conv2d(
            in_channels=hidden_shape,
            out_channels=hidden_shape*2,
            kernel_size=4,
            stride=2,
            padding=2)
        self.mp1 = torch.nn.MaxPool2d(2, stride=2)
        self.conv3 = torch.nn.Conv2d(
            in_channels=hidden_shape*2,
            out_channels=hidden_shape,
            kernel_size=3,
            stride=1,
            padding=1)
        self.mp2 = torch.nn.MaxPool2d(2, stride=2)
        self.fc = torch.nn.Linear(512, 128)

    def forward(self, feat):
        x = F.relu(self.conv1(feat))
        x = F.relu(self.conv2(x))
        x = self.mp1(x)
        x = F.relu(self.conv3(x))
        x = self.mp2(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc(x))
        return x


class Actor(torch.nn.Module):
    def __init__(self, extractor: Extractor, act_shape: int):
        super(Actor, self).__init__()

        self.extractor = extractor
        self.fc_out = torch.nn.Linear(128, act_shape)

    def forward(self, feat):
        x = self.extractor(feat)
        out = self.fc_out(x)
        return F.softmax(out, dim=1)


class Critic(torch.nn.Module):
    def __init__(self, extractor: Extractor, last_size: int = 1):
        super(Critic, self).__init__()

        self.extractor = extractor

        self.fc_out = torch.nn.Linear(128, last_size)

    def forward(self, feat):
        x = self.extractor(feat)
        out = self.fc_out(x)
        return out.squeeze(dim=-1)
