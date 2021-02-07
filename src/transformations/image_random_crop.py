
from numpy.random import randint
from torchvision.transforms.functional import crop


class ImageRandomCropFactory:

    def create(self, image_size: int, chunk_size: int, move_step: int, chunk_step: int, verbose: bool = False):
        assert chunk_size <= image_size
        move_ranges = dict()
        chunk_size_range = self.__create_range(chunk_size, image_size, chunk_step) if chunk_step != 0 else [chunk_size]

        for current_chunk_range in chunk_size_range:
            move_ranges[current_chunk_range] = [val - current_chunk_range for val in self.__create_range(current_chunk_range, image_size, move_step)]
            if verbose:
                print(f'ImageRandomCrop for {current_chunk_range} move range: {move_ranges} ')

        print('ImageRandomCrop chunk size range: ', chunk_size_range)
        return ImageRandomCrop(move_ranges, chunk_size_range)

    def __create_range(self, low: int, top: int, step: int):
        assert step > 0
        values_count = (top - low) // step + 1
        values = [(low + val * step) for val in range(values_count)]
        return values


class ImageRandomCrop:

    def __init__(self, move_ranges: dict, chunk_size_range: list):
        self.__move_ranges = move_ranges
        self.__chunk_size_range = chunk_size_range

    def get_move_range(self) -> list:
        return self.__move_ranges

    def get_chunk_size_range(self) -> list:
        return self.__chunk_size_range

    def __call__(self, img) -> tuple:
        chunk_index = randint(0, len(self.__chunk_size_range))
        current_chunk_size = self.__chunk_size_range[chunk_index]
        current_move_range = self.__move_ranges[current_chunk_size]

        move_indexes = randint(0, len(current_move_range), size=(2,))
        x_start = current_move_range[move_indexes[0]]
        y_start = current_move_range[move_indexes[1]]

        img = crop(img, y_start, x_start, current_chunk_size, current_chunk_size)

        return img, x_start, y_start, current_chunk_size
