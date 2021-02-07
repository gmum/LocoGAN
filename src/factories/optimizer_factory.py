import torch


def get_optimizers(generator_parameters, discriminator_parameters, hparams):
    generator_optimizer = torch.optim.Adam(generator_parameters, lr=hparams.lr, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator_parameters, lr=hparams.lr, betas=(0.5, 0.999))

    return [generator_optimizer, discriminator_optimizer]
