from __future__ import print_function
from os import makedirs
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from arg_parser import parse_args
from lightning_modules.locogan_module import LocoGanModule
from lightning_callbacks.sample_images_callback import SampleImagesCallback
from utils.noise_creator import NoiseCreator
from utils.images_from_chunks_creator import ImagesFromChunksCreator


def run():
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    print(f'Using random seed: {pl.seed_everything(args.random_seed)}')

    output_dir = f'../results/{args.chunk_size}of{args.image_size}/'
    makedirs(output_dir, exist_ok=True)
    print('Created output dir: ', output_dir)

    name = f'{args.lr}_{args.batch_size}'
    tb_logger = TensorBoardLogger(output_dir, name=name)

    callbacks = create_callbacks_and_evaluators(args)

    locogan_model = LocoGanModule(args)

    trainer = pl.Trainer(gpus=args.gpus,
                         max_epochs=args.max_epochs_count,
                         progress_bar_refresh_rate=20,
                         check_val_every_n_epoch=args.check_val_every_n_epoch,
                         default_root_dir=output_dir,
                         logger=tb_logger,
                         callbacks=callbacks,
                         log_every_n_steps=50)

    torch.cuda.reset_max_memory_allocated()
    trainer.fit(locogan_model)


def create_callbacks_and_evaluators(args) -> tuple:
    chunks_count = args.image_size // args.chunk_size
    full_face_noise_dim = args.noise_dim + (chunks_count - 1) * args.inner_noise_dim
    full_noise_creator = NoiseCreator(args.global_latent, args.local_latent, full_face_noise_dim)
    sampled_full_face_noise = full_noise_creator.create(64)
    images_from_chunks_creator = ImagesFromChunksCreator(args.chunk_size, args.image_size, args.inner_noise_dim, args.noise_dim)
    sample_images_callback = SampleImagesCallback(sampled_full_face_noise, images_from_chunks_creator)
    callbacks = [sample_images_callback]

    return callbacks


if __name__ == '__main__':
    run()
