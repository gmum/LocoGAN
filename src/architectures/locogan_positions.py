import torch.nn as nn
from modules.generator_block import GeneratorBlock
from modules.generator_output_block import GeneratorOutputBlock
from modules.discriminator_block import DiscriminatorBlock
from modules.discriminator_output_block import DiscriminatorOutputBlock
from modules.center_crop import CenterCrop
from modules.fast_positions_computer import FastPositionsComputer
from modules.position_appender import PositionAppender
from modules.unmap_tanh_zero_one import UnmapTanhZeroOne


class Generator(nn.Module):
    def __init__(self, input_channels: int, image_size: int, out_channels: int = 3):
        super(Generator, self).__init__()

        filters_base = 64
        self.__sequential_blocks = [
            # 10x10 (4x4)
            FastPositionsComputer(noise_border_width=3, image_size=image_size),
            PositionAppender(),
            # 10x10 (4x4)
            GeneratorBlock(input_channels + 2, filters_base*8, 4, 2, 3, output_size=14),
            # 14x14 (8x8)
            GeneratorBlock(filters_base*8, filters_base*4, 4, 2, 3, output_size=22),
            # 22x22 (16x16)
            GeneratorBlock(filters_base*4, filters_base*2, 4, 2, 3, output_size=38),
            # 38x38 (32x32)
            GeneratorBlock(filters_base*2, filters_base, 4, 2, 3, output_size=70),
            # 70x70 (64x64)
            GeneratorOutputBlock(filters_base, out_channels, 4, 1, 3),
            # 67x67 (64x64)
            CenterCrop(64)
        ]
        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, *input_data: tuple):
        assert len(input_data) == 4
        return self.main(input_data)


class Discriminator(nn.Module):
    def __init__(self, image_size: int):
        super(Discriminator, self).__init__()
        filters_base = 64
        self.main = nn.Sequential(
            UnmapTanhZeroOne(),
            FastPositionsComputer(0, image_size),
            PositionAppender(),
            DiscriminatorBlock(3 + 2, filters_base, 4, 2),
            DiscriminatorBlock(filters_base, filters_base*2, 4, 2),
            DiscriminatorBlock(filters_base*2, filters_base*4, 4, 2),
            DiscriminatorBlock(filters_base*4, filters_base*8, 4, 2),
            DiscriminatorOutputBlock(filters_base*8, 1, 4, 1)
        )

    def forward(self, *input_data):
        output = self.main(input_data)
        return output.view(-1, 1).squeeze(1)
