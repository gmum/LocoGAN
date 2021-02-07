from torch import Tensor
import torch.nn as nn


class CenterCrop(nn.Module):

    def __init__(self, size: int):
        super(CenterCrop, self).__init__()
        self.__size = size

    def forward(self, input_image: Tensor) -> Tensor:
        _, _, height, width = input_image.size()  # read in N, C, H, W

        assert height == width, "CenterCrop expects width to be equal to height"

        if height == self.__size:
            return input_image

        start_position = (height - self.__size)//2
        end_position = (height + self.__size)//2

        result = input_image[:, :, start_position:end_position, start_position:end_position]
        assert result.size(2) == self.__size and result.size(3) == self.__size, "Cropped size is not as expected"

        return result
