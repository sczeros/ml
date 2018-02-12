

#正则化 gluon

from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd as ag
import mxnet as mx

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt

num_train = 20
num_test = 100
num_inputs = 200

true_w = nd.ones((num_inputs, 1)) * 0.01
true_b = 0.05

x = nd.random_normal(shape = (num_train + num_test, num_inputs))
y = nd.dot(x, true_w) + true_b
y += 0.01 * nd.random_normal(shape = y.shape)

x_train, x_test = x[:num_train, :], x[num_train:, :]
y_train, y_test = y[:num_train], y[num_train:]





batch_size = 1
data_set_train = gluon.data.ArrayDataset(x_train, y_train)
train_data = gluon.data.DataLoader(data_set_train, batch_size, shuffle = True)

square_loss = gluon.loss.L2Loss()

net = gluon.nn.Sequential()

def cal_loss(net, x, y):
    return square_loss(net(x), y).mean().asscalar()

def train(weight_decay):
    epochs = 10
    learning_rate = 0.005
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(1))
    net.collect_params().initialize(mx.init.Normal(sigma = 1))

    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate' : learning_rate,
        'wd' : weight_decay
    })

    train_loss = []
    test_loss = []
    for e in range(epochs):
        for data, label in train_data:
            with ag.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
        train_loss.append(cal_loss(net, x_train, y_train))
        test_loss.append((cal_loss(net, x_test, x_test)))
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train', 'test'])
    plt.show()

    return ('learned w[:10]:', net[0].weight.data()[:, :10],
            'learned b:', net[0].bias.data())

print(train(0))