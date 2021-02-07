import torch
import torch.nn as nn


def unmap_tensor_function(tensor: torch.Tensor) -> torch.Tensor:
    return 2*(tensor - 0.5)


class UnmapTanhZeroOne(nn.Module):

    def forward(self, input_tuple: tuple) -> torch.Tensor:
        tensor, start_x, start_y, chunk_size = input_tuple
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0
        result = unmap_tensor_function(tensor)
        assert result.min() >= -1.0 and result.max() <= 1.0
        return (result, start_x, start_y, chunk_size)
