from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
from transformations.image_random_crop import ImageRandomCropFactory
from transformations.image_resize import ImageResize
from transformations.image_to_tensor import ImageToTensor


def get_dataloader(args) -> DataLoader:
    image_size = args.image_size
    chunk_size = args.chunk_size
    move_step = args.move_step
    chunk_step = args.chunk_step

    dataset_transforms = []

    if args.resize_dataset:
        print('Adding resize initial image step')
        dataset_transforms += [
            transforms.Resize(args.image_size),
        ]

    dataset_transforms += [
        ImageRandomCropFactory().create(image_size, chunk_size, move_step, chunk_step)
    ]

    if args.chunk_step != 0:
        print('Adding resize chunk step')
        dataset_transforms += [
            ImageResize(chunk_size)
        ]

    dataset_transforms += [
        ImageToTensor()
    ]

    dataset_transforms = transforms.Compose(dataset_transforms)

    dataset = datasets.ImageFolder(root=args.dataroot,
                                   transform=dataset_transforms)

    assert dataset
    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True,
                            shuffle=True, num_workers=args.workers)
    return dataloader

def get_plain_dataloader(args):
    dataset = datasets.ImageFolder(root=args.dataroot,
                                   transform=transforms.Compose([ToTensor()]))

    assert dataset
    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True,
                            shuffle=True, num_workers=args.workers)
    return dataloader
