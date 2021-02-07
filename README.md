# Repository info

This repository contains an implementation of LocoGAN - Locally Convolutional GAN) in PyTorch, proposed by [Łukasz Struski, Szymon Knop, Jacek Tabor, Wiktor Daniec, Przemysław Spurek (2020)](https://arxiv.org/abs/2002.07897).

# Contents of the repository

```
|-- src/ - contains an implementation of LocoGAN allowing to reproduce experiments from the original paper
|---- architectures/ - files containing architectures proposed in the paper
|---- factories/ - factories used to create objects proper objects base on command line arguments
|---- lightning_callbacks/ - directory contains PyTorch Lightning callbacks used during the experiments
|---- lightning_modules/ - directory contains PyTorch Lightning modules to conduct the experiments
|---- modules/ - custom neural network layers used in models
|---- tests/ - a bunch of unit tests
|---- train_locogan.py - the main script to run all of the experiments
|-- results/ - directory that will be created to store the results of conducted experiments
|-- data/ - default directory that will be used as a source of data and place to download datasets
```

Experiments are written in `pytorch-lightning` to decouple the science code from the engineering. For more details refer to [PyTorch-Lightning documentation](https://github.com/PyTorchLightning/pytorch-lightning)

## Conducting the experiments

The simplest experiment can be executed by running the following command in the `src` directory:

`python -m train_locogan --dataroot <path_to_ffhq_thumbnails_dataset>`

## Other options

The code allows manipulating some of the parameters(for example using other versions of the model, changing learning rate values) for more info see the list of available arguments in `src/args_parser.py` file

To run the unit tests execute the following command:
`python -m unittest`

# Environment

- python3
- pytorch
- torchvision
- numpy
- pytorch-lightning

# Additional links

FID Score can be computed with the following code and repository:

- https://github.com/bioinf-jku/TTUR

For the experiments presented in this repository we used the [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset.

# License

This implementation is licensed under the MIT License
