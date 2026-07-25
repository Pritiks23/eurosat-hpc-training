import time

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler
)


import config

from dataset import create_dataloaders

from model import EuroSATNet

from utils import (
    seed_everything,
    save_checkpoint,
    calculate_accuracy,
    get_gpu_memory,
    AverageMeter
)



# ============================================================
# Optional CUDA preprocessing extension
# ============================================================


try:

    import native_ops

    USE_NATIVE_OPS = True


except ImportError:

    print(
        "native_ops unavailable. "
        "Using PyTorch preprocessing."
    )

    USE_NATIVE_OPS = False





# ============================================================
# CUDA preprocessing wrapper
# ============================================================


def preprocess(images):

    """
    Runs custom CUDA image normalization.

    Input:

        [batch, channels, height, width]


    Output:

        normalized tensor

    """

    if (
        USE_NATIVE_OPS
        and
        images.is_cuda
    ):

        images = native_ops.normalize_images(
            images,
            config.CUDA_MEAN,
            config.CUDA_STD,
            True
        )[0]


    return images





# ============================================================
# Validation
# ============================================================


def validate(
        model,
        loader,
        criterion
):


    model.eval()


    loss_meter = AverageMeter()

    accuracy_meter = AverageMeter()



    with torch.no_grad():


        for images, labels in loader:


            images = images.to(
                config.DEVICE,
                non_blocking=True
            )


            labels = labels.to(
                config.DEVICE,
                non_blocking=True
            )


            outputs = model(images)



            loss = criterion(
                outputs,
                labels
            )



            accuracy = calculate_accuracy(
                outputs,
                labels
            )



            loss_meter.update(
                loss.item(),
                images.size(0)
            )


            accuracy_meter.update(
                accuracy,
                images.size(0)
            )



    return (
        loss_meter.average,
        accuracy_meter.average
    )





# ============================================================
# Training
# ============================================================


def train():

    seed_everything(
        config.SEED
    )


    writer = SummaryWriter(
        log_dir=config.RUN_DIR
    )



    train_loader, val_loader, _ = (
        create_dataloaders()
    )



    model = EuroSATNet(
        config.NUM_CLASSES
    )


    model = model.to(
        config.DEVICE
    )



    criterion = nn.CrossEntropyLoss()



    optimizer = optim.AdamW(

        model.parameters(),

        lr=config.LEARNING_RATE,

        weight_decay=config.WEIGHT_DECAY

    )



    scheduler = optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=config.EPOCHS

    )



    scaler = torch.cuda.amp.GradScaler(
        enabled=config.USE_AMP
    )



    best_accuracy = 0.0



    print(
        f"Training on {config.DEVICE}"
    )



    # ========================================================
    # PyTorch Profiler
    # ========================================================


    with profile(

        activities=[

            ProfilerActivity.CPU,

            ProfilerActivity.CUDA

        ],


        schedule=schedule(

            wait=1,

            warmup=1,

            active=3

        ),


        on_trace_ready=
        tensorboard_trace_handler(

            str(
                config.RUN_DIR /
                "profiler"
            )

        ),


        record_shapes=True,

        profile_memory=True

    ) as prof:



        for epoch in range(
            config.EPOCHS
        ):


            model.train()



            loss_meter = AverageMeter()

            accuracy_meter = AverageMeter()



            progress = tqdm(

                train_loader,

                desc=
                f"Epoch {epoch}"

            )



            for batch_idx, (
                images,
                labels
            ) in enumerate(progress):


                images = images.to(

                    config.DEVICE,

                    non_blocking=True

                )


                labels = labels.to(

                    config.DEVICE,

                    non_blocking=True

                )



                # ------------------------
                # CUDA preprocessing
                # ------------------------


                images = preprocess(
                    images
                )



                optimizer.zero_grad(
                    set_to_none=True
                )



                # ------------------------
                # Mixed precision forward
                # ------------------------


                with torch.cuda.amp.autocast(

                    enabled=config.USE_AMP

                ):


                    outputs = model(
                        images
                    )


                    loss = criterion(

                        outputs,

                        labels

                    )



                # ------------------------
                # Backpropagation
                # ------------------------


                scaler.scale(
                    loss
                ).backward()



                scaler.step(
                    optimizer
                )


                scaler.update()



                accuracy = calculate_accuracy(

                    outputs,

                    labels

                )



                loss_meter.update(

                    loss.item(),

                    images.size(0)

                )


                accuracy_meter.update(

                    accuracy,

                    images.size(0)

                )



                progress.set_postfix({

                    "loss":
                    f"{loss_meter.average:.3f}",

                    "acc":
                    f"{accuracy_meter.average:.3f}"

                })



                prof.step()



            # =================================================
            # Validation after epoch
            # =================================================


            val_loss, val_accuracy = validate(

                model,

                val_loader,

                criterion

            )



            scheduler.step()



            print(

                f"""

Epoch {epoch}

Train Loss:
{loss_meter.average:.4f}

Train Accuracy:
{accuracy_meter.average:.4f}

Validation Loss:
{val_loss:.4f}

Validation Accuracy:
{val_accuracy:.4f}

"""

            )



            # TensorBoard

            writer.add_scalar(

                "Loss/train",

                loss_meter.average,

                epoch

            )


            writer.add_scalar(

                "Loss/validation",

                val_loss,

                epoch

            )


            writer.add_scalar(

                "Accuracy/train",

                accuracy_meter.average,

                epoch

            )


            writer.add_scalar(

                "Accuracy/validation",

                val_accuracy,

                epoch

            )



            if torch.cuda.is_available():


                memory = get_gpu_memory()



                writer.add_scalar(

                    "GPU/memory_allocated",

                    memory["allocated"],

                    epoch

                )



            # Save latest checkpoint


            save_checkpoint(

                model,

                optimizer,

                scheduler,

                epoch,

                val_loss,

                val_accuracy,

                config.LAST_CHECKPOINT_PATH

            )



            # Save best model


            if val_accuracy > best_accuracy:


                best_accuracy = val_accuracy



                torch.save(

                    model.state_dict(),

                    config.BEST_MODEL_PATH

                )



                print(
                    "Saved new best model"
                )



    writer.close()



if __name__ == "__main__":

    train()
