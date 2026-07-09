from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader

BATCH_SIZE = 64


def load_dataset():
    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
        root="../data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root="../data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = load_dataset()

    images,labels = next(iter(train_loader))

    print(images.shape)
    print(labels.shape)
