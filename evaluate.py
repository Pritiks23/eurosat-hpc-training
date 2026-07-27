""" This file evaluates the trained model on the unseen test dataset to measure how well it generalizes to new images. 
It loads the saved best model, performs inference on the test set without computing gradients, and calculates performance
metrics such as accuracy, precision, recall, F1 score, the confusion matrix, and a classification report. 
These metrics are saved to a JSON file and printed to summarize the model's overall classification performance after training."""
import json

import torch

import numpy as np

from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import config

from dataset import create_dataloaders

from model import EuroSATNet





def evaluate():

    """
    Runs final evaluation on the
    held-out test set.

    This answers:

    Did the model actually learn?
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


    model = model.to(
        config.DEVICE
    )


    model.eval()



    all_predictions = []

    all_labels = []



    with torch.no_grad():


        for images, labels in tqdm(

            test_loader,

            desc="Evaluating"

        ):


            images = images.to(
                config.DEVICE
            )


            labels = labels.to(
                config.DEVICE
            )



            outputs = model(
                images
            )


            predictions = torch.argmax(

                outputs,

                dim=1

            )



            all_predictions.extend(

                predictions.cpu()
                .numpy()

            )


            all_labels.extend(

                labels.cpu()
                .numpy()

            )



    # Convert to numpy

    y_true = np.array(
        all_labels
    )


    y_pred = np.array(
        all_predictions
    )



    # Metrics

    accuracy = accuracy_score(

        y_true,

        y_pred

    )


    precision = precision_score(

        y_true,

        y_pred,

        average="macro"

    )


    recall = recall_score(

        y_true,

        y_pred,

        average="macro"

    )


    f1 = f1_score(

        y_true,

        y_pred,

        average="macro"

    )



    cm = confusion_matrix(

        y_true,

        y_pred

    )



    report = classification_report(

        y_true,

        y_pred,

        target_names=config.CLASS_NAMES,

        output_dict=True

    )



    metrics = {


        "accuracy":

            float(accuracy),


        "precision_macro":

            float(precision),


        "recall_macro":

            float(recall),


        "f1_macro":

            float(f1),


        "confusion_matrix":

            cm.tolist(),


        "classification_report":

            report

    }



    output_file = (

        config.METRIC_DIR /

        "evaluation_metrics.json"

    )



    with open(

        output_file,

        "w"

    ) as f:


        json.dump(

            metrics,

            f,

            indent=4

        )



    print(
        "\nEvaluation Complete"
    )


    print(
        f"Accuracy: {accuracy:.4f}"
    )


    print(
        f"F1 Score: {f1:.4f}"
    )


    print(
        f"Saved metrics to {output_file}"
    )



    return metrics





if __name__ == "__main__":

    evaluate()
