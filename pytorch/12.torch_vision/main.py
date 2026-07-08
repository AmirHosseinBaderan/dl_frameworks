from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

images, labels = next(iter(loader))
print(images.shape)
print(labels.shape)

transforms.Resize((224,224))

plt.title(labels[0])
plt.imshow(images[0][0],cmap="gray")
plt.show()

transforms.Normalize(
    mean=(0.5,),
    std=(0.5,)
)

transform = transforms.Compose([

    transforms.Resize((32,32)),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5,),
        (0.5,)
    )

])