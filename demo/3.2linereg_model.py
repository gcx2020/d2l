import torch
import random
from d2l import torch as d2l


def synthetic_data(w, b, nums_examples):
    """generate y = Xw + b + noise"""
    X = torch.normal(0, 1, (nums_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.1, y.shape)
    return X, y.reshape(-1, 1)


def data_iter(batch_size, labels, features):
    """yield mini-batch data"""
    num_samples = len(labels)
    indices = list(range(num_samples))
    random.shuffle(indices)
    for i in range(0, num_samples, batch_size):
        index = torch.tensor(indices[i : min(i + batch_size, num_samples)])
        yield features[index], labels[index]


def linear_model(X, w, b):
    """linear model :y = xw + b"""
    return torch.matmul(X, w) + b


def loss(y_hat, y):
    """loss function : l = (y_hat - y)^2 / 2"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def init_params():
    """initialize w and b"""
    w = torch.normal(0, 0.1, size=(2, 1), requires_grad=True)
    b = torch.normal(0, 0.1, size=(1, 1), requires_grad=True)
    return w, b


def sgd(params, lr, batch_size):
    """mini-batch stochastic gradient descent"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size


if __name__ == "__main__":
    real_w = torch.tensor([2, -3.4])
    real_b = 4.2
    features, labels = synthetic_data(real_w, real_b, 1000)
    d2l.set_figsize()
    d2l.plt.scatter(features[:, 1].detach().numpy(), labels[:].detach().numpy(), 1)
    d2l.plt.savefig("3.2linereg_model.svg")

    for X, y in data_iter(10, labels, features):
        print(X, y)
        break
    init_w, init_b = init_params()
    lr = 0.01
    epochs = 10
    batch_size = 10
    for epoch in range(epochs):
        for X, y in data_iter(batch_size, labels, features):
            l = loss(linear_model(X, init_w, init_b), y)
            for param in [init_w, init_b]:  # 梯度清零
                if param.grad is not None:
                    param.grad.data.zero_()
            l.sum().backward()  # 批量数据迭代，需要叠加loss,然后去求梯度
            sgd([init_w, init_b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(linear_model(features, init_w, init_b), labels)
            print(f"epoch:{epoch},loss:{train_l.mean()}")

    print(f"w的估计误差: {real_w - init_w.reshape(real_w.shape)}")
    print(f"b的估计误差: {real_b - init_b}")
