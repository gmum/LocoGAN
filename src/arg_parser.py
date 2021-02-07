import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate used for the experiments')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--move_step', type=int, default=4, help='')
    parser.add_argument('--chunk_step', type=int, default=4, help='')
    parser.add_argument('--image_size', type=int, default=128, help='the height / width of the whole image from the dataset')
    parser.add_argument('--chunk_size', type=int, default=64, help='the height / width of the input sub-image to network')
    parser.add_argument('--n_disc', type=int, default=1, help='describes how many discriminator steps for one generator step')
    parser.add_argument('--local_latent', type=int, default=2, help='how many local latent channels')
    parser.add_argument('--global_latent', type=int, default=16, help='how many global latent channels')
    parser.add_argument('--noise_dim', type=int, default=10, help='what is the size of image-like noise')
    parser.add_argument('--inner_noise_dim', type=int, default=4, help='what is the size of image-like noise without padding')
    parser.add_argument('--max_epochs_count', type=int, default=10000, help='limits training numbers of epochs')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5, help='')
    parser.add_argument('--measure_memory', action='store_true', help='adds information about max_memory_allocated to logs')
    parser.add_argument('--resize_dataset', action='store_true', help='flag configures dataloader to resize dataset images to image_size.')
    parser.add_argument('--gpus', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--random_seed', type=int, default=59876312, help='manual seed')

    opt = parser.parse_args()
    return opt
