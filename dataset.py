import torch

from torchvision import transforms
from torchvision.datasets import EuroSAT

from torch.utils.data import (
    DataLoader,
    random_split
)

import config



def get_transforms():

    """
    Image preprocessing pipeline.

    EuroSAT images:
        - RGB
        - 64x64 pixels

    Output:
        Tensor shape:
        [3,64,64]

    """

    return transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                config.NORMALIZE_MEAN,
                config.NORMALIZE_MEAN,
                config.NORMALIZE_MEAN
            ],

            std=[
                config.NORMALIZE_STD,
                config.NORMALIZE_STD,
                config.NORMALIZE_STD
            ]
        )

    ])



def load_eurosat():

    """
    Downloads and loads EuroSAT.

    Dataset:
        ~27,000 satellite images
        10 land-cover classes

    """

    dataset = EuroSAT(

        root=config.DATA_DIR,

        download=True,

        transform=get_transforms()

    )

    return dataset



def split_dataset(dataset):

    """
    Creates:

        70% train
        15% validation
        15% test

    The generator seed ensures
    reproducible experiments.
    """

    total_size = len(dataset)


    train_size = int(
        config.TRAIN_SPLIT *
        total_size
    )


    val_size = int(
        config.VAL_SPLIT *
        total_size
    )


    test_size = (
        total_size
        -
        train_size
        -
        val_size
    )


    train_set, val_set, test_set = random_split(

        dataset,

        [
            train_size,
            val_size,
            test_size
        ],

        generator=torch.Generator()
        .manual_seed(config.SEED)

    )


    return (
        train_set,
        val_set,
        test_set
    )



def create_dataloaders():

    """
    Creates PyTorch DataLoaders.

    Training:
        shuffle=True

    Validation/Test:
        deterministic ordering
    """

    dataset = load_eurosat()


    train_set, val_set, test_set = (
        split_dataset(dataset)
    )


    train_loader = DataLoader(

        train_set,

        batch_size=config.BATCH_SIZE,

        shuffle=True,

        num_workers=config.NUM_WORKERS,

        pin_memory=config.PIN_MEMORY,

        persistent_workers=True

    )


    val_loader = DataLoader(

        val_set,

        batch_size=config.BATCH_SIZE,

        shuffle=False,

        num_workers=config.NUM_WORKERS,

        pin_memory=config.PIN_MEMORY,

        persistent_workers=True

    )


    test_loader = DataLoader(

        test_set,

        batch_size=config.BATCH_SIZE,

        shuffle=False,

        num_workers=config.NUM_WORKERS,

        pin_memory=config.PIN_MEMORY,

        persistent_workers=True

    )


    return (

        train_loader,

        val_loader,

        test_loader

    )
