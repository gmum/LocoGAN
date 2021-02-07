from argparse import Namespace

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

from factories.architecture_factory import get_architecture
from factories.dataloader_factory import get_dataloader
from factories.optimizer_factory import get_optimizers

from utils.noise_creator import NoiseCreator


class LocoGanModule(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = Namespace(**args)

        self.__noise_channels = args.local_latent + args.global_latent
        self.__generator, self.__discriminator = get_architecture(self.__noise_channels, args.image_size)
        self.__chunk_noise_creator = NoiseCreator(args.global_latent, args.local_latent, args.noise_dim)
        self.__gen_images: torch.FloatTensor = None

        self.__random_sampled_noise: torch.Tensor = None
        self.__criterion = nn.BCELoss()
        self.hparams = args

        self.real_label = 1
        self.fake_label = 0

    def get_generator(self) -> torch.nn.Module:
        return self.__generator

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return get_optimizers(self.__generator.parameters(), self.__discriminator.parameters(), self.hparams)

    def train_dataloader(self) -> DataLoader:
        return get_dataloader(self.hparams)

    def training_epoch_end(self, outputs):
        if self.hparams.measure_memory:
            max_memory_allocated = torch.cuda.max_memory_allocated()
            self.log('max_memory_allocated', max_memory_allocated)
            print(f'Max memory allocated: {max_memory_allocated}')
            torch.cuda.reset_max_memory_allocated()

    def forward(self, noise_with_positions: torch.Tensor) -> tuple:
        generated_images = self.__generator(noise_with_positions)
        discriminator_output = self.__discriminator(generated_images)
        return generated_images, discriminator_output

    def loss_function(self, images: torch.FloatTensor, start_x: torch.FloatTensor, start_y: torch.FloatTensor, chunk_size: torch.FloatTensor, target_label: torch.FloatTensor):
        disc_output = self.__discriminator(images, start_x, start_y, chunk_size)
        return self.__criterion(disc_output, target_label).mean()

    def generate_images(self, batch_size: int, start_x: torch.FloatTensor, start_y: torch.FloatTensor, chunk_size: torch.FloatTensor):
        gen_input = self.__chunk_noise_creator.create(batch_size, self.device)
        self.__gen_images = self.__generator(gen_input, start_x, start_y, chunk_size)

    def __generator_training_step(self, batch_size: int, start_x: torch.FloatTensor, start_y: torch.FloatTensor, chunk_size: torch.FloatTensor) -> dict:
        self.generate_images(batch_size, start_x, start_y, chunk_size)
        reals = torch.full((batch_size,), self.real_label, dtype=torch.float, device=self.device)
        g_loss = self.loss_function(self.__gen_images, start_x, start_y, chunk_size, reals)
        self.log('g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True)
        return g_loss

    def __discriminator_training_step(self, batch_size: int, real_img: torch.FloatTensor, start_x: torch.FloatTensor, start_y: torch.FloatTensor, chunk_size: torch.FloatTensor) -> dict:
        reals = torch.full((batch_size,), self.real_label, dtype=torch.float, device=self.device)
        fakes = torch.full((batch_size,), self.fake_label, dtype=torch.float, device=self.device)

        real_loss = self.loss_function(real_img, start_x, start_y, chunk_size, reals)
        fake_loss = self.loss_function(self.__gen_images.detach(), start_x, start_y, chunk_size, fakes)

        d_loss = (real_loss + fake_loss)/2.0

        self.log('d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True)
        return d_loss

    def training_step(self, batch: tuple, batch_index: int, optimizer_idx: int) -> dict:
        mini_batch = batch[0]
        start_x = mini_batch[1].to(self.device, non_blocking=True)
        start_y = mini_batch[2].to(self.device, non_blocking=True)
        chunk_size = mini_batch[3].to(self.device, non_blocking=True)
        batch_size = mini_batch[0].size(0)
        is_generator_step = batch_index % self.hparams.n_disc == 0

        # Generator
        if optimizer_idx == 0 and is_generator_step:
            return self.__generator_training_step(batch_size, start_x, start_y, chunk_size)

        # Discriminator
        if optimizer_idx == 1:
            if not is_generator_step:
                with torch.no_grad():
                    self.generate_images(batch_index, start_x, start_y, chunk_size)
            real_img = mini_batch[0].to(self.device, non_blocking=True)
            return self.__discriminator_training_step(batch_size, real_img, start_x, start_y, chunk_size)

        raise ValueError("Invalid optimizer_idx")
