import torch
import torch.nn as nn
from torchvision import models


class XGazeNetwork(nn.Module):
    """ResNet50 + linear head that matches ETH-XGaze baseline structure."""

    def __init__(self) -> None:
        super().__init__()
        self.gaze_network = models.resnet50(weights=None)
        self.gaze_fc = nn.Sequential(nn.Linear(2048, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep layer flow explicit to match ETH-XGaze style backbone usage.
        x = self.gaze_network.conv1(x)
        x = self.gaze_network.bn1(x)
        x = self.gaze_network.relu(x)
        x = self.gaze_network.maxpool(x)
        x = self.gaze_network.layer1(x)
        x = self.gaze_network.layer2(x)
        x = self.gaze_network.layer3(x)
        x = self.gaze_network.layer4(x)
        feature = self.gaze_network.avgpool(x)
        feature = torch.flatten(feature, 1)
        gaze = self.gaze_fc(feature)
        return gaze
