import os
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.cuda.amp import autocast, GradScaler

from dataset import create_dataloaders

import tensorboardX


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 20
LR = 1e-3


os.makedirs("checkpoints", exist_ok=True)


print("=" * 60)
print("EuroSAT GPU Training Pipeline")
print("=" * 60)

print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


writer = tensorboardX.SummaryWriter(
    "./runs/eurosat_training"
)


# ============================================================
# MODEL
# ============================================================


class EuroSATNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(),


            nn.AdaptiveAvgPool2d((1,1))
        )


        self.classifier = nn.Sequential(

            nn.Dropout(0.3),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                10
            )

        )


    def forward(self,x):

        x = self.features(x)

        x = x.flatten(1)

        return self.classifier(x)



# ============================================================
# DATA
# ============================================================


print("\nLoading EuroSAT dataset...")


train_loader, val_loader, test_loader = create_dataloaders()


print(
    "Train:",
    len(train_loader.dataset)
)

print(
    "Validation:",
    len(val_loader.dataset)
)

print(
    "Test:",
    len(test_loader.dataset)
)



# ============================================================
# TRAINING SETUP
# ============================================================


model = EuroSATNet().to(DEVICE)


criterion = nn.CrossEntropyLoss()


optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)


scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


scaler = GradScaler()



# ============================================================
# TRAIN LOOP
# ============================================================


best_accuracy = 0



for epoch in range(EPOCHS):


    model.train()


    running_loss = 0

    correct = 0

    total = 0



    start = time.time()



    for images, labels in train_loader:


        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad()


        with autocast():

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )


        scaler.scale(loss).backward()

        scaler.step(
            optimizer
        )

        scaler.update()



        running_loss += loss.item()



        predictions = torch.argmax(
            outputs,
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)



    train_accuracy = correct / total

    train_loss = running_loss / len(train_loader)



    scheduler.step()



    # ========================================================
    # VALIDATION
    # ========================================================


    model.eval()


    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:


            images = images.to(DEVICE)

            labels = labels.to(DEVICE)


            outputs = model(images)


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            val_correct += (
                predictions == labels
            ).sum().item()


            val_total += labels.size(0)



    val_accuracy = val_correct / val_total



    elapsed = time.time() - start



    print(
        f"""
Epoch {epoch+1}/{EPOCHS}

Train Loss:
{train_loss:.4f}

Train Accuracy:
{train_accuracy:.4f}

Validation Accuracy:
{val_accuracy:.4f}

Time:
{elapsed:.2f}s
"""
    )



    writer.add_scalar(
        "Loss/train",
        train_loss,
        epoch
    )


    writer.add_scalar(
        "Accuracy/train",
        train_accuracy,
        epoch
    )


    writer.add_scalar(
        "Accuracy/validation",
        val_accuracy,
        epoch
    )



    torch.save(

        {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "accuracy":
                val_accuracy
        },

        f"checkpoints/checkpoint_epoch_{epoch}.pt"

    )



    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy


        torch.save(
            model.state_dict(),
            "best_model.pth"
        )



# ============================================================
# FINAL SAVE
# ============================================================


torch.save(
    model.state_dict(),
    "final_model_weights.pth"
)



with open(
    "training_metrics.json",
    "w"
) as f:

    json.dump(
        {
            "best_validation_accuracy":
                best_accuracy
        },

        f,

        indent=4
    )



writer.close()



print("\nTraining Complete!")

print(
    "Best Validation Accuracy:",
    best_accuracy
)

print(
    "Saved:"
)

print(
    "- best_model.pth"
)

print(
    "- final_model_weights.pth"

)

print(
    "- checkpoints/"
)
