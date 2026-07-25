import matplotlib.pyplot as plt

import torch

import numpy as np

from torchvision import transforms

from dataset import create_dataloaders

from model import EuroSATNet

import config





def denormalize(image):

    """
    Converts normalized tensors
    back into displayable images.

    Model input:

        normalized RGB tensor

    Visualization:

        RGB image [0,1]

    """

    image = image.cpu().numpy()

    image = np.transpose(
        image,
        (1,2,0)
    )


    image = (
        image *
        config.NORMALIZE_STD
        +
        config.NORMALIZE_MEAN
    )


    image = np.clip(
        image,
        0,
        1
    )


    return image





def visualize_predictions(
        num_images=12
):

    """
    Creates a prediction grid.

    Shows:

        Image

        True label

        Predicted label

        Confidence

    """



    _, _, test_loader = (
        create_dataloaders()
    )



    model = EuroSATNet(
        config.NUM_CLASSES
    )


    model.load_state_dict(

        torch.load(

            config.BEST_MODEL_PATH,

            map_location=config.DEVICE

        )

    )


    model.to(
        config.DEVICE
    )


    model.eval()



    images, labels = next(

        iter(test_loader)

    )


    images = images[:num_images]

    labels = labels[:num_images]



    with torch.no_grad():


        outputs = model(

            images.to(
                config.DEVICE
            )

        )


        probabilities = torch.softmax(

            outputs,

            dim=1

        )


        confidence, predictions = torch.max(

            probabilities,

            dim=1

        )



    rows = 3

    cols = 4



    plt.figure(

        figsize=(14,10)

    )



    for idx in range(num_images):


        plt.subplot(

            rows,

            cols,

            idx + 1

        )



        plt.imshow(

            denormalize(

                images[idx]

            )

        )



        prediction = (

            config.CLASS_NAMES[

                predictions[idx].item()

            ]

        )



        actual = (

            config.CLASS_NAMES[

                labels[idx].item()

            ]

        )



        score = (

            confidence[idx].item()
            *
            100

        )



        correct = (

            prediction ==
            actual

        )


        title = (

            f"Pred: {prediction}\n"

            f"True: {actual}\n"

            f"{score:.1f}%"

        )


        if correct:

            title += "\n✓"

        else:

            title += "\n✗"



        plt.title(

            title,

            fontsize=9

        )


        plt.axis(
            "off"
        )



    plt.tight_layout()



    output = (

        config.RESULT_DIR /

        "prediction_grid.png"

    )


    plt.savefig(

        output,

        dpi=300

    )


    print(

        f"Saved visualization to {output}"

    )





if __name__ == "__main__":

    visualize_predictions()
