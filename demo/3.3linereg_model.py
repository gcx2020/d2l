import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data


def synthetic_data(w, b, nums_examples):
    """generate y = Xw + b + noise"""
    X = torch.normal(0, 1.0, (nums_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.1, y.shape)
    return X, y.reshape(-1, 1)


def data_load(data_array, batch_size, is_train=True):  # 生成器函数
    """generate mini-batch data by nn.utils.data"""
    dataset = data.TensorDataset(*data_array)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


if __name__ == "__main__":
    # initialize parameters
    real_w = torch.tensor([2, -3.4])
    real_b = 4.2
    batch_size = 10
    epochs = 10

    # generate synthetic data
    features, labels = synthetic_data(real_w, real_b, 1000)

    # data loader by batch size
    data_iter = data_load((features, labels), batch_size, is_train=True)
    print(next(iter(data_iter)))

    # define linear model
    net = nn.Sequential(nn.Linear(2, 1))

    # initialize liner model parameters
    net[0].weight.data.normal_(0, 0.1)
    net[0].bias.data.fill_(0.0)

    # define loss function
    loss = nn.MSELoss()  # mean squared error loss/loss function

    # define optimizer SGD
    trainer = optim.SGD(params=net.parameters(), lr=0.03)

    # start training
    for epoch in range(epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()  # clear accumulated gradient
            l.backward()  # back propagation
            trainer.step()  # update parameters
        l = loss(net(features), labels)
        print(f"epoch {epoch + 1}, loss {l.mean().item()}")
    print(f"w误差:", real_w - net[0].weight.data.reshape(real_w.shape))
    print(f"b误差:", real_b - net[0].bias.data)
