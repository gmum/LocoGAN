import unittest
import torch
from utils.noise_creator import NoiseCreator


class TestNoiseCreator(unittest.TestCase):

    def test_simple_scenario(self):
        # Arrange
        uut = NoiseCreator(global_latent=2, local_latent=1, noise_dim=4)

        # Act
        noise = uut.create(2)

        # Assert
        self.assertAlmostEqual(noise[0][0].sum().item(), noise[0][0][0][0].item() * 16, places=3)
        self.assertAlmostEqual(noise[0][1].sum().item(), noise[0][1][0][0].item() * 16, places=3)
        self.assertNotEqual(noise[0][2][0][1].item(), noise[0][2][0][0].item())
        self.assertNotEqual(noise[0][2][1][1].item(), noise[0][2][1][0].item())

        self.assertAlmostEqual(noise[1][0].sum().item(), noise[1][0][0][0].item() * 16, places=3)
        self.assertAlmostEqual(noise[1][1].sum().item(), noise[1][1][0][0].item() * 16, places=3)
        self.assertNotEqual(noise[1][2][0][1].item(), noise[1][2][0][0].item())
        self.assertNotEqual(noise[1][2][1][1].item(), noise[1][2][1][0].item())
