import torch
import torch.nn as nn
from modules.unmap_tanh_zero_one import unmap_tensor_function


class FastPositionsComputer(nn.Module):

    def __init__(self, noise_border_width: int, image_size: int):
        super(FastPositionsComputer, self).__init__()
        self.__noise_border_width = noise_border_width
        self.__image_size = image_size

    def forward(self, tensor_tuple: tuple) -> tuple:
        input_data, start_x, start_y, chunk_size = tensor_tuple
        device = start_x.device
        batch_size, _, target_size, _ = input_data.size()

        inner_noise_dim = target_size - 2 * self.__noise_border_width
        border_width = self.__noise_border_width * chunk_size / inner_noise_dim
        start_x = start_x.view(batch_size, 1, 1, 1)
        start_y = start_y.view(batch_size, 1, 1, 1)

        total_chunk_size = chunk_size + 2*border_width
        step = total_chunk_size / target_size

        base_vector = torch.arange(target_size, dtype=torch.float, device=device) * step.view(-1, 1) - border_width.view(-1, 1) + 0.5 + (step.view(-1, 1) - 1) / 2

        base_vector = base_vector.repeat(1, target_size).view(batch_size, 1, target_size, target_size)
        x_positions = base_vector + start_x
        y_positions = base_vector.transpose(2, 3) + start_y

        positions = torch.cat((y_positions, x_positions), 1) / self.__image_size
        positions = unmap_tensor_function(positions)

        return input_data, positions.detach()
