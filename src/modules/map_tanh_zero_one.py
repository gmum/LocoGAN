import torch
import torch.nn as nn


class MapTanhZeroOne(nn.Module):

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.min() >= -1.0 and tensor.max() <= 1.0
        return tensor / 2.0 + 0.5
