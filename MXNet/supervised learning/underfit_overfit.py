

#欠拟合和过拟合

from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd as ag

#y=1.2x−3.4x^2+5.6x^3+5.0+noise
#噪音服从 均值为0 标准差为0.1的正态分布 or 高斯分布
true_w = [1.2, -3.4, 5.6]
true_b = 5.0

num_train = 100
num_test = 100

# x = nd.random_normal(shape = (num_train + num_test, 3))
# print(x[0])
#[ 2.21220636  0.7740038   1.04344046]
x = nd.random_normal(shape = (num_train + num_test, 1))
x = nd.concat(x, nd.power(x, 2), nd.power(x, 3))
print(x[0])
#[  2.21220636   4.893857    10.82622147]

y = true_w[0] * x[:, 0] + true_w[1] * x[:, 1] + true_w[2] * x[:, 2] + true_b
y += 0.1 * nd.random_normal(shape = y.shape)

print(x[:5], y[:5])

#定义训练和测试步骤

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt

def train(x_train, x_test, y_train, y_test):
    #线性模型
    net = gluon.nn.Sequential()
    with ag.record():
        net.add(gluon.nn.Dense(1))
    net.initialize()

    #设置一些默认参数
    epochs = 100
    learning_rate = 0.01
    batch_size = min(10, y_train.shape[0])
    dataset_train = gluon.data.ArrayDataset(x_train, y_train)
    data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle = True)

    #默认SGD和均方误差
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : learning_rate})
    square_loss = gluon.loss.L2Loss()

    #保存训练和测试误差
    train_loss = []
    test_loss = []
    for epoch in range(epochs):
        for data, label in data_iter_train:
            with ag.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()

            trainer.step(batch_size)
            train_loss.append(square_loss(net(x_train), y_train).mean().asscalar())
            test_loss.append(square_loss(net(x_test), y_test).mean().asscalar())

    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train', 'test'])
    plt.show()

    return ('learned weight', net[0].weight.data(),
            'learned bias', net[0].bias.data())

result = train(x[:num_train,], x[num_test:,], y[:num_train], y[num_train:])
print(result)
