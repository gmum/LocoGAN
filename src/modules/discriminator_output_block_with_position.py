import torch
import torch.nn as nn
from modules.position_appender import PositionAppender
from modules.position_upsampler import PositionUpsampler
from modules.discriminator_output_block import DiscriminatorOutputBlock


class DiscriminatorOutputBlockWithPosition(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DiscriminatorOutputBlockWithPosition, self).__init__()
        sequential_blocks = [
            PositionUpsampler(),
            PositionAppender(),
            DiscriminatorOutputBlock(in_channels + 2, out_channels, kernel_size, stride)
        ]

        self.block = nn.Sequential(*sequential_blocks)

    def forward(self, input_tuple: tuple) -> torch.Tensor:
        return self.block(input_tuple)
