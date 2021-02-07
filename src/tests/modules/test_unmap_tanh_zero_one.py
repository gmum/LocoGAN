import unittest
import torch
from modules.unmap_tanh_zero_one import UnmapTanhZeroOne
from modules.map_tanh_zero_one import MapTanhZeroOne


class TestUnmapTanhZeroOne(unittest.TestCase):

    def test_correctly_unmaps_random_image_and_leaves_positions_unchanged(self):
        # Arrange
        test_input_image = torch.rand(64, 3, 64, 64)
        start_x = torch.randn(1)
        start_y = torch.randn(1)
        chunk_size = torch.randint(0, 64, size=(1,))
        uut = UnmapTanhZeroOne()

        # Act
        result = uut((test_input_image, start_x, start_y, chunk_size))

        # Assert
        self.assertEqual(4, len(result))
        self.assertLessEqual(result[0].max().item(), 1)
        self.assertGreaterEqual(result[0].min().item(), -1)
        self.assertEqual(result[1], start_x)
        self.assertEqual(result[2], start_y)
        self.assertEqual(result[3], chunk_size)

    def test_correctly_unmaps_image(self):
        # Arrange
        test_input_image = torch.FloatTensor([
            [
                [
                    [0.0, 0.5],
                    [0.1, 1.0]
                ]
            ]
        ])
        start_x = torch.FloatTensor([0.1])
        start_y = torch.FloatTensor([2.1])
        chunk_size = torch.FloatTensor([64])
        uut = UnmapTanhZeroOne()

        # Act
        result = uut((test_input_image, start_x, start_y, chunk_size))

        # Assert
        result_image = result[0]
        self.assertEqual(4, len(result))
        self.assertLessEqual(result_image[0][0][0][0], -1)
        self.assertLessEqual(result_image[0][0][0][1], 0)
        self.assertLessEqual(result_image[0][0][1][0], -0.8)
        self.assertLessEqual(result_image[0][0][1][1], 1.0)
        self.assertEqual(result[1], 0.1)
        self.assertEqual(result[2], 2.1)

    def test_is_neutral_with_mapping(self):
        # Arrange
        test_input_image = 2 * torch.rand(64, 3, 64, 64) - 1  # Creates uniform [-1, +1] vector
        start_x = torch.randn(1)
        start_y = torch.randn(1)
        chunk_size = torch.randint(0, 64, size=(1,))

        uut1 = MapTanhZeroOne()
        uut2 = UnmapTanhZeroOne()

        # Act
        mapped_image = uut1(test_input_image)
        result_image, _, _, _ = uut2((mapped_image, start_x, start_y, chunk_size))

        # Assert
        self.assertTrue(torch.all(torch.eq(result_image, test_input_image)))
