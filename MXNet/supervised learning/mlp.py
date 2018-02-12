

#Multi Layer Perceptron
#MLP

#多层神经网络,至少包括一个隐层

from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd as ag

def SGD(params, learning_rate):
    for param in params:
        param[:] = param - learning_rate * param.grad

#继续使用Fashionmnist 数据集

def transform(data, label):
    return data.astype('float32') / 225, label.astype('float32')

mnist_train = gluon.data.vision.FashionMNIST(train = True, transform = transform)
mnist_test = gluon.data.vision.FashionMNIST(train = False, transform = transform)

batch_size = 256

train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle = True)
test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle = False)

#多层感知机 与 逻辑回归非常的相似,不过是多了一个隐层,在输入层
#和输出层之间

num_inputs = 784 # 28 * 28
num_outputs = 10
num_hidden = 256

weight_scale = 0.01

#隐层与输入层之间的权重矩阵
w1 = nd.random_normal(shape = (num_inputs, num_hidden), scale = weight_scale)
#偏置向量
b1 = nd.random_normal(shape = num_hidden)

#输出层与隐层之间的权重矩阵
w2 = nd.random_normal(shape = (num_hidden, num_outputs), scale = weight_scale)
b2 = nd.random_normal(shape = num_outputs)

params = [w1, b1 , w2, b2]

for param in params:
    param.attach_grad()

#激活函数
#整流线性单元
#Rectified Linear Unit 线性整流函数
def relu(X):
    return nd.maximum(X, 0)

#也可使用sigmoid激活函数
def sigmoid(X):
    return 1 / (1 + nd.exp(X))

#定义模型

#将层全连接起来 和 激活函数(RELU)串接起来

# def net(X):
#     X = X.reshape((-1, num_inputs))
#     h1 = relu(nd.dot(X, w1) + b1)
#     output = nd.dot(h1, w2) + b2
#     return output

#使用sigmoid 的神经网络 或 说多层感知机
def net(X):
    X = X.reshape((-1, num_inputs))
    h1 = relu(nd.dot(X, w1) + b1)
    output = nd.dot(h1, w2) + b2
    return output

def accuracy(output, label):
    return nd.mean(output.argmax(axis = 1) == label).asscalar()

def evaluate_accuracy(data_iterator, net):
    acc = 0.
    for data, label in data_iterator:
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_iterator)
#损失函数
#softmax and crossEntropy
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

#开始训练

learning_rate = .5
epochs = 5

for epoch in range(epochs):
    train_loss = 0.
    train_acc = 0.

    for data, label in train_data:
        with ag.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)

        loss.backward()
        SGD(params, learning_rate / batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    test_acc = evaluate_accuracy(test_data, net)

    print('Epoch %d, Train loss %f, Train acc %f, Test acc %f.' % (
        epoch, train_loss / len(train_data), train_acc / len(train_data), test_acc
    ))


def get_text_labels(label):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress,', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in label]

data, label = mnist_test[0:9]
print('ture label')
print(get_text_labels(label))

predicted_labels = net(data).argmax(axis=1)
print('predict label')
print(get_text_labels(predicted_labels.asnumpy()))