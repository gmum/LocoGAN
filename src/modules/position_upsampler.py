import torch.nn as nn


class PositionUpsampler(nn.Module):

    def forward(self, tensor_tuple: tuple) -> tuple:
        input_image, positions = tensor_tuple
        _, _, height, width = input_image.size()
        upsampled_positions = nn.Upsample(size=(height, width), mode='bilinear', align_corners=False)(positions)
        return input_image, upsampled_positions
