import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim=2, hidden=64, dropout_p=0.2):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.drop1 = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop2 = nn.Dropout(dropout_p)
        self.out = nn.Linear(hidden, action_dim)

        # small initial output weights -> start near zero thrust
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)

        # input scaling: obs is groups of 4 = [position(2), velocity(2)]
        scale = torch.tensor([10.0, 10.0, 2.0, 2.0] * (obs_dim // 4))
        self.register_buffer("obs_scale", scale, persistent=False)

    def forward(self, x):
        x = x / self.obs_scale
        x = torch.relu(self.fc1(x))
        x = self.drop1(x)
        x = torch.relu(self.fc2(x))
        x = self.drop2(x)
        return torch.tanh(self.out(x))


def mc_dropout_predict(net, obs_tensor, n_samples=20):
    net.train()  # IMPORTANT: keeps dropout active during inference
    with torch.no_grad():
        samples = torch.stack([net(obs_tensor) for _ in range(n_samples)])
    mean_action = samples.mean(dim=0)
    uncertainty = samples.std(dim=0).mean(dim=-1)
    return mean_action, uncertainty