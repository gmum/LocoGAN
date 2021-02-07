import torch
import torch.nn as nn
from modules.center_crop import CenterCrop


class GeneratorBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_size):
        super(GeneratorBlock, self).__init__()

        sequential_blocks = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                               stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CenterCrop(output_size)
        ]

        self.main = nn.Sequential(*sequential_blocks)

    def forward(self, input_tuple: torch.Tensor) -> torch.Tensor:
        return self.main(input_tuple)
