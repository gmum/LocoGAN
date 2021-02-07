import unittest
import torch
from torchvision.transforms import ToPILImage, ToTensor
from transformations.image_random_crop import ImageRandomCrop


class TestImageRandomCrop(unittest.TestCase):

    def test_correctly_crops_image_01(self):
        # Arrange
        test_input_image_tensor = torch.zeros((3, 128, 128))
        test_input_image_tensor[0:3, 32:96, 32:96] = torch.ones(3, 64, 64)
        test_input_image = ToPILImage(mode='RGB')(test_input_image_tensor)

        uut = ImageRandomCrop({64: [32]}, [64])

        # Act
        output_image, start_x, start_y, chunk_size = uut(test_input_image)
        output_tensor = ToTensor()(output_image)

        # Assert
        self.assertTrue(torch.all(torch.eq(output_tensor[0], torch.ones(1, 3, 64, 64))))
        self.assertEqual(start_x, 32)
        self.assertEqual(start_y, 32)
        self.assertEqual(chunk_size, 64)

    def test_correctly_crops_image_10(self):
        # Arrange
        test_input_image_tensor = torch.ones((3, 128, 128))
        test_input_image_tensor[0:3, 32:96, 32:96] = torch.zeros(3, 64, 64)
        test_input_image = ToPILImage(mode='RGB')(test_input_image_tensor)

        uut = ImageRandomCrop({64: [32]}, [64])

        # Act
        output_image, start_x, start_y, chunk_size = uut(test_input_image)
        output_tensor = ToTensor()(output_image)

        # Assert
        self.assertTrue(torch.all(torch.eq(output_tensor[0], torch.zeros(1, 3, 64, 64))))
        self.assertEqual(start_x, 32)
        self.assertEqual(start_y, 32)
        self.assertEqual(chunk_size, 64)
