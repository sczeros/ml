

#使用 gluon 实现逻辑回归
from mxnet import ndarray as nd
from mxnet import autograd as ag
from mxnet import gluon


def SGD(params, learning_rate):
    for param in params:
        param[:] = param[:] - learning_rate * param.grad


def transform(data, label):
    return data.astype('float32') / 225, label.astype('float32')

mnist_train = gluon.data.vision.FashionMNIST(train = True, transform = transform)
mnist_test = gluon.data.vision.FashionMNIST(train =False, transform = transform )

batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle = True)
test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle = False)

#定义初始化模型
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(10))

net.initialize()

#softmax和交叉熵函数
softmax_corss_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

#优化
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : 0.1})

#精度
def accuracy(output, label):
    return nd.mean(output.argmax(axis = 1) == label).asscalar()

#评估模型

def evaluate_accuracy(data_iterator, net):
    acc = 0.
    for data, label in data_iterator:
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_iterator)


#训练
epochs = 5

for epoch in range(epochs):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with ag.record():
            output = net(data)
            loss = softmax_corss_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    test_acc = evaluate_accuracy(test_data, net)
    print("Epoch %d, Train loss %f, Train acc %f, Test acc %f." % (
        epoch, train_loss / len(train_data), train_acc / len(train_data), test_acc
    ))
