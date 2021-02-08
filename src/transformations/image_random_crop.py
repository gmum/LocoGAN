
from numpy.random import randint
from torchvision.transforms.functional import crop


class ImageRandomCropFactory:

    def create(self, image_size: int, chunk_size: int, verbose: bool = False):
        assert chunk_size <= image_size
        move_ranges = dict()
        move_ranges = [val - chunk_size for val in self.__create_range(chunk_size, image_size, chunk_size)]
        print(f'ImageRandomCrop for {chunk_size} move range: {move_ranges}')
        return ImageRandomCrop(move_ranges, chunk_size)

    def __create_range(self, low: int, top: int, step: int):
        assert step > 0
        values_count = (top - low) // step + 1
        values = [(low + val * step) for val in range(values_count)]
        return values


class ImageRandomCrop:

    def __init__(self, move_ranges: dict, chunk_size: int):
        self.__move_ranges = move_ranges
        self.__chunk_size = chunk_size

    def __call__(self, img) -> tuple:
        move_indexes = randint(0, len(self.__move_ranges), size=(2,))
        x_start = self.__move_ranges[move_indexes[0]]
        y_start = self.__move_ranges[move_indexes[1]]

        img = crop(img, y_start, x_start, self.__chunk_size, self.__chunk_size)
        return img, x_start, y_start, self.__chunk_size
