import torch
import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision import transforms

from model import FashionClassifier


MODEL_PATH = "../models/fashion_classifier.pth"


class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


def predict():

    model = FashionClassifier()

    model.load_state_dict(
        torch.load(MODEL_PATH)
    )

    model.eval()

    transform = transforms.ToTensor()

    test_dataset = datasets.FashionMNIST(
        root="../data",
        train=False,
        download=True,
        transform=transform
    )

    image, label = test_dataset[0]

    input_image = image.unsqueeze(0)

    with torch.no_grad():

        outputs = model(input_image)

        _, predicted = torch.max(
            outputs,
            dim=1
        )

    predicted = predicted.item()

    print(
        "Prediction:",
        class_names[predicted]
    )

    print(
        "Actual:",
        class_names[label]
    )

    plt.imshow(
        image.squeeze().numpy(),
        cmap="gray"
    )

    plt.title(
        class_names[predicted]
    )

    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    predict()