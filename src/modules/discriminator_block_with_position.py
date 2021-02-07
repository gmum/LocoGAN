import torch.nn as nn
from modules.position_appender import PositionAppender
from modules.position_upsampler import PositionUpsampler
from modules.discriminator_block import DiscriminatorBlock


class DiscriminatorBlockWithPosition(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DiscriminatorBlockWithPosition, self).__init__()

        sequential_blocks = [
            PositionUpsampler(),
            PositionAppender(),
            DiscriminatorBlock(in_channels + 2, out_channels, kernel_size, stride)
        ]

        self.block = nn.Sequential(*sequential_blocks)

    def forward(self, input_tuple: tuple) -> tuple:
        _, positions = input_tuple
        output = self.block(input_tuple)
        return output, positions
