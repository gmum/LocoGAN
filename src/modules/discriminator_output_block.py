import torch
import torch.nn as nn


class DiscriminatorOutputBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DiscriminatorOutputBlock, self).__init__()
        sequential_blocks = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0),
            nn.Sigmoid()
        ]

        self.block = nn.Sequential(*sequential_blocks)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.block(input_tensor)
