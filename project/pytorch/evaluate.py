import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

from dataset import load_dataset
from model import FashionClassifier


MODEL_PATH = "../models/fashion_classifier.pth"


def evaluate():

    _, test_loader = load_dataset()

    model = FashionClassifier()

    model.load_state_dict(
        torch.load(MODEL_PATH)
    )

    model.eval()

    correct = 0
    total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                dim=1
            )

            correct += (
                predicted == labels
            ).sum().item()

            total += labels.size(0)

            y_true.extend(
                labels.numpy()
            )

            y_pred.extend(
                predicted.numpy()
            )

    accuracy = correct / total

    print(f"Test Accuracy : {accuracy:.4f}")

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.show()


if __name__ == "__main__":
    evaluate()