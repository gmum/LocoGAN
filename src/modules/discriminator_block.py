import torch
import torch.nn as nn


class DiscriminatorBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DiscriminatorBlock, self).__init__()

        sequential_blocks = [
            nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        self.block = nn.Sequential(*sequential_blocks)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.block(input_tensor)
