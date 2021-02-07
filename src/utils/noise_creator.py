import torch


class NoiseCreator():

    def __init__(self, global_latent: int, local_latent: int, noise_dim: int):
        self.__global_latent = global_latent
        self.__local_latent = local_latent
        self.__noise_dim = noise_dim

    def create(self, batch_size: int, device: str = None):

        local_latent = torch.randn(batch_size, self.__local_latent, self.__noise_dim, self.__noise_dim, device=device)
        if self.__global_latent == 0:
            return local_latent

        global_latent = torch.randn(batch_size, self.__global_latent, 1, 1, device=device).expand(-1, -1, self.__noise_dim, self.__noise_dim)
        return torch.cat((global_latent, local_latent), 1)
