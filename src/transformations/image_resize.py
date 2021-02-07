
from torchvision.transforms.functional import resize


class ImageResize:

    def __init__(self, resize_target: int):
        self.__resize_target = resize_target

    def __call__(self, data_tuple) -> tuple:
        img = resize(data_tuple[0], (self.__resize_target, self.__resize_target))
        return img, data_tuple[1], data_tuple[2], data_tuple[3]
