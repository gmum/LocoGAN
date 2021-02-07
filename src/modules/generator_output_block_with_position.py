import torch.nn as nn
from modules.position_appender import PositionAppender
from modules.fast_positions_computer import FastPositionsComputer
from modules.generator_output_block import GeneratorOutputBlock


class GeneratorOutputBlockWithPosition(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 noise_border_width: int, image_size: int):
        super(GeneratorOutputBlockWithPosition, self).__init__()

        sequential_blocks = [
            FastPositionsComputer(noise_border_width, image_size),
            PositionAppender(),
            GeneratorOutputBlock(in_channels + 2, out_channels, kernel_size,
                                 stride, padding=padding),
        ]

        self.main = nn.Sequential(*sequential_blocks)

    def forward(self, input_tuple: tuple) -> tuple:
        return self.main(input_tuple)
