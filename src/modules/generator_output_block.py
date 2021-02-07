import torch
import torch.nn as nn
from modules.map_tanh_zero_one import MapTanhZeroOne


class GeneratorOutputBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(GeneratorOutputBlock, self).__init__()

        sequential_blocks = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                               stride, padding=padding, bias=True),
            nn.Tanh(),
            MapTanhZeroOne()
        ]

        self.main = nn.Sequential(*sequential_blocks)

    def forward(self, input_tuple: torch.Tensor) -> torch.Tensor:
        return self.main(input_tuple)
