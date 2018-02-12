
#正则化

from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd as ag

#y=0.05+∑i=1p0.01xi+noise


num_train = 20
num_test = 100
num_inputs = 200

true_w = nd.ones((num_inputs, 1)) * 0.01
true_b = 0.05

x = nd.random_normal(shape = (num_train + num_test, num_inputs))
y = nd.dot(x, true_w) + true_b
y += 0.1 * nd.random_normal(shape = y.shape)
x_train, x_test = x[:num_train,], x[num_train:,]
y_train, y_test = y[:num_train], x[num_train:]

import random
batch_size = 1
def data_iter(num_example):
    idx = list(range(num_example))
    random.shuffle(idx)
    for i in range(0, num_example, batch_size):
        j = nd.array(idx[i:min(i + batch_size, num_example)])
        yield x.take(j), y.take(j)
       # yield nd.take(x, j), nd.take(y, j)

scale = 1

def init_param():
    w = nd.random_normal(shape = (num_inputs, 1), scale = scale)
    b = nd.zeros(shape = (1,))
    params = [w, b]
    for param in params:
        param.attach_grad()
    return params

#L2 范数正则化

def l2_penalty(w, b):
    return ((w ** 2).sum() + b ** 2) / 2


#%matplotlib inline
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt

def net(data, w, b):
    return nd.dot(data, w) + b

def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2 / 2

def SGD(params, learning_rate, batch_size):
    for param in params:
        param[:] = param - learning_rate * param.grad / batch_size

def cal_loss(net, params, x, y):
    return square_loss(net(x, *params), y).mean().asscalar()

def train(lambdax):
    epochs = 10
    learning_rate = 0.005
    w, b = params = init_param()
    train_loss = []
    test_loss= []
    for epoch in range(epochs):
        for data, label in data_iter(num_train):
            with ag.record():
                output = net(data, *params)
                loss = square_loss(output, label) + lambdax * l2_penalty(*params)
            loss.backward()
            SGD(params, learning_rate, batch_size)

            train_loss.append(cal_loss(net, params, x_train, y_train))
            test_loss.append(cal_loss(net, params, x_test, y_test))
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train', 'test'])
    plt.show()
    return 'learned w[:10] : ' , w[:10].T , 'learned b' , b

print(train(5))
