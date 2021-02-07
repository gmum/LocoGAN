from pytorch_lightning import Trainer, Callback
import torch
from utils.images_from_chunks_creator import ImagesFromChunksCreator
from lightning_modules.locogan_module import LocoGanModule


class SampleImagesCallback(Callback):
    def __init__(self, input_noise: torch.Tensor, images_from_chunks_creator: ImagesFromChunksCreator):
        super().__init__()
        self.__images_from_chunks_creator = images_from_chunks_creator
        self.__input_noise = input_noise

    def on_epoch_end(self, trainer: Trainer, pl_module: LocoGanModule):
        if trainer.current_epoch % trainer.check_val_every_n_epoch != 0:
            return

        # generate images
        with torch.no_grad():
            pl_module.eval()
            generator = pl_module.get_generator()
            input_noise = self.__input_noise.to(pl_module.device)
            sampled_images = self.__images_from_chunks_creator.get_images(generator, input_noise, pl_module.device)
            pl_module.train()

        trainer.logger.experiment.add_images('sampled_images', sampled_images, trainer.current_epoch)
