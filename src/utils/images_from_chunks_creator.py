import torch


class ImagesFromChunksCreator():

    def __init__(self, chunk_size: int, image_size: int, inner_noise_dim: int, noise_dim: int):
        self.__chunk_size = chunk_size
        self.image_size = image_size
        self.__inner_noise_dim = inner_noise_dim
        self.__noise_dim = noise_dim

    def get_images(self, generator: torch.nn.Module, noise: torch.Tensor, device: str) -> torch.Tensor:
        samples_count = noise.size(0)

        result_images = torch.zeros(samples_count, 3, self.image_size, self.image_size, device=device)
        chunks_count = self.image_size // self.__chunk_size
        chunk_sizes = self.__chunk_size * torch.ones((samples_count, 1), dtype=torch.float, device=device)
        for h_chunk_index in range(chunks_count):
            x_start_noise = h_chunk_index * self.__inner_noise_dim
            x_end_noise = x_start_noise + self.__noise_dim
            x_pos = h_chunk_index*self.__chunk_size
            fullface_x = torch.ones((samples_count, 1), dtype=torch.float, device=device) * x_pos

            for v_chunk_index in range(chunks_count):
                y_pos = v_chunk_index*self.__chunk_size
                y_start_noise = v_chunk_index * self.__inner_noise_dim
                y_end_noise = y_start_noise + self.__noise_dim
                current_noise = noise[:, :, y_start_noise:y_end_noise, x_start_noise:x_end_noise]
                fullface_y = torch.ones((samples_count, 1), dtype=torch.float, device=device) * y_pos
                current_chunk = generator(current_noise, fullface_x, fullface_y, chunk_sizes)
                result_images[:, :, y_pos:(y_pos+self.__chunk_size), x_pos:(x_pos+self.__chunk_size)] = current_chunk.detach()

        return result_images
