
from torchvision.transforms.functional import to_tensor


class ImageToTensor:

    def __call__(self, data_tuple) -> tuple:
        return to_tensor(data_tuple[0]), data_tuple[1], data_tuple[2], data_tuple[3]
