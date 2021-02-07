from architectures.locogan_positions import Generator, Discriminator


def get_architecture(input_channels: int, image_size: int):
    generator = Generator(input_channels, image_size)
    generator.apply(weights_init)
    discriminator = Discriminator(image_size)
    discriminator.apply(weights_init)
    return generator, discriminator

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
