import torch.nn as nn
from modules.position_appender import PositionAppender
from modules.fast_positions_computer import FastPositionsComputer
from modules.generator_block import GeneratorBlock


class GeneratorBlockWithPosition(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 noise_border_width: int, image_size, output_size):
        super(GeneratorBlockWithPosition, self).__init__()

        sequential_blocks = [
            FastPositionsComputer(noise_border_width, image_size),
            PositionAppender(),
            GeneratorBlock(in_channels + 2, out_channels, kernel_size,
                           stride, padding=padding, output_size=output_size)
        ]

        self.main = nn.Sequential(*sequential_blocks)

    def forward(self, input_tuple: tuple) -> tuple:
        _, starting_x, starting_y, chunk_size = input_tuple
        output = self.main(input_tuple)
        return output, starting_x, starting_y, chunk_size
