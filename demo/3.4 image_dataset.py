import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils import data
from typing import List
from d2l import torch as d2l


def get_fashion_mnist_labels(labels: List[int]):
    """covert labels to text"""
    labels_text = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]
    return [
        labels_text[int(index)] for index in labels if 0 <= index < len(labels_text)
    ]


def show_images(imags, num_rows, num_cols, title=None, scale=1.5):
    """plot images"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axle = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axles = axle.flatten()
    for i, (ax, img) in enumerate(zip(axles, imags)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        if title is not None:
            ax.set_title(title[i])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    return axle


def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]  # convert image to tensor
    if resize is not None:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        "./data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        "./data", train=False, transform=trans, download=True
    )
    return torch.utils.data.DataLoader(
        dataset=mnist_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=get_dataloader_workers(),
    ), torch.utils.data.DataLoader(
        dataset=mnist_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=get_dataloader_workers(),
    )


def get_dataloader_workers():
    """多线程读取数据"""
    return 7


if __name__ == "__main__":
    # data preparation
    trans = transforms.ToTensor()  # convert image to tensor
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=trans
    )
    print(f"train data size: {len(mnist_train)}, test data size: {len(mnist_test)}" "")
    print(f"data shape: {mnist_train[0][0].shape}")

    # data load
    X, y = next(iter(data.DataLoader(mnist_train, batch_size=18, shuffle=True)))

    # show images
    show_images(X.reshape(18, -1, 28), 2, 9, title=get_fashion_mnist_labels(y))
    plt.savefig("3.4image_dataset.svg")

    # muti task data load
    batch_size = 256
    data_iter = torch.utils.data.DataLoader(
        dataset=mnist_train,
        shuffle=True,
        batch_size=batch_size,
        num_workers=get_dataloader_workers(),
    )

    # test data load speed
    timer = d2l.Timer()
    for X, y in data_iter:
        continue
    print(f"{len(mnist_train) / timer.stop():.2f} examples/sec")

    # test all components
    train_iter, test_iter = load_data_fashion_mnist(batch_size=256, resize=64)
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break
    for X, y in test_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break
