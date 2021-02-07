import unittest
import torch
from modules.position_upsampler import PositionUpsampler


class TestPositionUpsampler(unittest.TestCase):

    def test_correctly_downsamples_positions(self):
        # Arrange
        x_positions = [
            [0., 1., 2., 3., 4., 5., 6., 7.],
            [0., 1., 2., 3., 4., 5., 6., 7.],
            [0., 1., 2., 3., 4., 5., 6., 7.],
            [0., 1., 2., 3., 4., 5., 6., 7.],
            [0., 1., 2., 3., 4., 5., 6., 7.],
            [0., 1., 2., 3., 4., 5., 6., 7.],
            [0., 1., 2., 3., 4., 5., 6., 7.],
            [0., 1., 2., 3., 4., 5., 6., 7.],
        ]
        y_positions = [
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1.],
            [2., 2., 2., 2., 2., 2., 2., 2.],
            [3., 3., 3., 3., 3., 3., 3., 3.],
            [4., 4., 4., 4., 4., 4., 4., 4.],
            [5., 5., 5., 5., 5., 5., 5., 5.],
            [6., 6., 6., 6., 6., 6., 6., 6.],
            [7., 7., 7., 7., 7., 7., 7., 7.],
        ]
        positions = torch.FloatTensor([
            [
                x_positions,
                y_positions
            ]
        ])

        test_input_image = torch.randn((1, 3, 4, 4))
        uut = PositionUpsampler()

        # Act
        output_image, upsampled_positions = uut((test_input_image, positions))

        # Assert
        self.assertTrue(torch.all(torch.eq(output_image, test_input_image)))
        self.assertEqual(4, upsampled_positions.size(2))
        self.assertEqual(4, upsampled_positions.size(3))

        expected_positions = torch.FloatTensor([[[[0.5, 2.5, 4.5, 6.5],
                                                  [0.5, 2.5, 4.5, 6.5],
                                                  [0.5, 2.5, 4.5, 6.5],
                                                  [0.5, 2.5, 4.5, 6.5]],

                                                 [[0.5, 0.5, 0.5, 0.5],
                                                  [2.5, 2.5, 2.5, 2.5],
                                                  [4.5, 4.5, 4.5, 4.5],
                                                  [6.5, 6.5, 6.5, 6.5]]]])
        self.assertTrue(torch.all(torch.eq(expected_positions, upsampled_positions)))

    def test_correctly_downsamples_positions_other_part(self):
        # Arrange
        x_positions = [
            [8., 9., 10., 11., 12., 13., 14., 15.],
            [8., 9., 10., 11., 12., 13., 14., 15.],
            [8., 9., 10., 11., 12., 13., 14., 15.],
            [8., 9., 10., 11., 12., 13., 14., 15.],
            [8., 9., 10., 11., 12., 13., 14., 15.],
            [8., 9., 10., 11., 12., 13., 14., 15.],
            [8., 9., 10., 11., 12., 13., 14., 15.],
            [8., 9., 10., 11., 12., 13., 14., 15.],
        ]
        y_positions = [
            [6., 6., 6., 6., 6., 6., 6., 6.],
            [7., 7., 7., 7., 7., 7., 7., 7.],
            [8., 8., 8., 8., 8., 8., 8., 8.],
            [9., 9., 9., 9., 9., 9., 9., 9.],
            [10., 10., 10., 10., 10., 10., 10., 10.],
            [11., 11., 11., 11., 11., 11., 11., 11.],
            [12., 12., 12., 12., 12., 12., 12., 12.],
            [13., 13., 13., 13., 13., 13., 13., 13.],
        ]
        positions = torch.FloatTensor([
            [
                x_positions,
                y_positions
            ]
        ])

        test_input_image = torch.randn((1, 3, 2, 2))
        uut = PositionUpsampler()

        # Act
        output_image, upsampled_positions = uut((test_input_image, positions))

        # Assert
        self.assertTrue(torch.all(torch.eq(output_image, test_input_image)))
        self.assertEqual(2, upsampled_positions.size(2))
        self.assertEqual(2, upsampled_positions.size(3))
        expected_positions = torch.FloatTensor([[[[9.5, 13.5],
                                                  [9.5, 13.5]],

                                                 [[7.5, 7.5],
                                                  [11.5, 11.5]]]])
        self.assertTrue(torch.all(torch.eq(expected_positions, upsampled_positions)))
