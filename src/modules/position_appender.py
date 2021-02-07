import torch
import torch.nn as nn


class PositionAppender(nn.Module):

    def forward(self, tensor_tuple: tuple) -> tuple:
        input_data, positions = tensor_tuple
        input_with_added_position = torch.cat((input_data, positions), 1)
        return input_with_added_position
