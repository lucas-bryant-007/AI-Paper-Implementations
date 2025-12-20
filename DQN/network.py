import torch
from torch import nn


class QNetwork(nn.Module):
    """
    Classic DQN CNN for 84x84 inputs.
    Input shape: (B, C=4, 84, 84)
    Output shape: (B, n_actions)
    """

    def __init__(self, n_actions: int, input_channels: int = 4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        # 84 -> 20 -> 9 -> 7, so feature map is (64, 7, 7)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)
