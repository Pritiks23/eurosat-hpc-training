import random
import numpy as np
import torch

from pathlib import Path



def seed_everything(seed=42):
    """
    Makes experiments reproducible.

    Controls:
        - Python random
        - NumPy
        - PyTorch CPU
        - PyTorch CUDA

    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


    # Deterministic CUDA behavior

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False





def save_checkpoint(
        model,
        optimizer,
        scheduler,
        epoch,
        loss,
        accuracy,
        path
):
    """
    Saves complete training state.

    Allows:

        python train.py --resume

    without restarting.

    Stores:

        - model weights
        - optimizer state
        - LR scheduler state
        - epoch
        - metrics

    """


    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "loss": loss,

        "accuracy": accuracy

    }


    torch.save(
        checkpoint,
        path
    )






def load_checkpoint(
        path,
        model,
        optimizer=None,
        scheduler=None
):
    """
    Loads previous training state.

    Returns:
        starting epoch
        previous metrics

    """


    checkpoint = torch.load(
        path,
        map_location="cpu"
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    if optimizer:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )


    if scheduler:

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )


    return (
        checkpoint["epoch"],
        checkpoint["loss"],
        checkpoint["accuracy"]
    )






def calculate_accuracy(
        outputs,
        labels
):
    """
    Computes batch accuracy.

    Example:

        Predictions:
        [Forest, River, Highway]

        Labels:
        [Forest, Forest, Highway]


        Accuracy:
        2/3

    """


    predictions = torch.argmax(
        outputs,
        dim=1
    )


    correct = (
        predictions == labels
    ).sum().item()


    total = labels.size(0)


    return correct / total





def get_gpu_memory():
    """
    Returns current GPU memory usage.

    Useful for TensorBoard logging.

    """

    if not torch.cuda.is_available():

        return {
            "allocated": 0,
            "reserved": 0
        }


    return {

        "allocated":
            torch.cuda.memory_allocated()
            /
            (1024 ** 3),


        "reserved":
            torch.cuda.memory_reserved()
            /
            (1024 ** 3)

    }





class AverageMeter:
    """
    Tracks running averages.

    Example:

        Loss:
        1.5
        1.2
        0.9

        Average:
        1.2

    """

    def __init__(self):

        self.reset()



    def reset(self):

        self.value = 0

        self.total = 0

        self.count = 0

        self.average = 0



    def update(
            self,
            value,
            n=1
    ):

        self.value = value

        self.total += value * n

        self.count += n

        self.average = (
            self.total /
            self.count
        )
