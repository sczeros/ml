
#多类逻辑回归

from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd as ag
from matplotlib import pyplot as plt

def transform(data, label):
    return data.astype('float32')/225, label.astype('float32')


#获取数据,处理数据过程
mnist_train = gluon.data.vision.FashionMNIST(train = True, transform = transform)
mnist_test = gluon.data.vision.FashionMNIST(train = False, transform = transform)

data,label = mnist_train[0]
#print('example shape: ', data.shape, 'label:', label)
#print(mnist_train.__len__())
def show_image(images):
    n = images.shape[0]
    _,figs = plt.subplots(1, n, figsize=(15,15))
    for i in range(n):
        figs[i].imshow(images[i].reshape((28,28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)

    plt.show()

def get_text_labels(label):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress,', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in label]

data,label = mnist_train[0:9]
#show_image(data)
# print(label)
# print(get_text_labels(label))

#数据读取

#虽然我们可以像前面那样通过yield来定义获取批量数据函数，
#这里我们直接使用gluon.data的DataLoader函数，它每次yield一个批量。
batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

#每次从训练数据中取得一个随机的样本组成的批量,但测试数据不必如此
#print(train_data)

#初始化模型参数
#逻辑回归

#跟线性模型一样,每个样本会表示成一个向量,这里图片的 shape (28,28) ,则输入
#向量的长度为 28*28 = 784. 因为我们要做的是多分类,需要做的是对每一个类别需要
#预测这个样本属于此类的概率.因为类型总计有十种,所以输出向量的长度为10
num_outputs = 10
num_inputs = 784

#权重为 784*10 的矩阵
w = nd.random_normal(shape = (num_inputs, num_outputs))
b = nd.random_normal(shape = num_outputs)

params = [w,b]
#print(params)


#赋予梯度 还不是很理解


for param in params:
    param.attach_grad()



#定义模型
#在线性回归里,只需要输出一个标量yhat 使得尽可能的靠近目标值.
#但在逻辑回归中,我们需要每个类别的概率,这些概率值为正,而且加起来为1.
#通常的做法是通过softmax函数来将任意的输入归一化为合法的任意值
#from mxnet import nd
def softmax(X):
    exp = nd.exp(X)
    partition = exp.sum(axis = 1, keepdims = True)
    return exp / partition


# x = nd.random_normal(shape=(1,10))
# print(x)
# x_prob = softmax(x)
# print(x_prob)
# print(x_prob.sum(axis = 1))

#模型

def net(X):
    return softmax(nd.dot(X.reshape((-1,num_inputs)), w) + b)

#交叉损失函数
#需要定义一个针对预测概率的损失函数.
#将两个概率的分布的交叉熵值作为目标值,最小化这个值等价于最大化这两个概率的相似度

#需要先将真实标号表示成一个分布,例如如果y=1,那么其对应的分布就是除了第二个元素为1
#,其他元素全为0的一个长度为10的标量,也就是yvec = [0,1,0,0,0,0,0,0,0,0]
#那么交叉熵就是yvec[0]*log(yhat[0])+...+yvec[n]*log(yhat[n]).
# 注意到yvec里面只有一个1,那么前面等价于log(yhat[y]).
# 所以我们可以定义这个损失函数了
def cross_entropy(yhat, y):
    return -nd.pick(nd.log(yhat), y)

#计算精度
#给定一个概率输出,将输出概率最高的那个类作为作为预测的类,然后通过比较真实标记
#可以计算精度

def accuracy(output, label):
    return nd.mean(output.argmax(axis = 1) == label).asscalar()

#评估模型
def evaluate_accuracy(data_iterator, net):
    acc = 0.
    for data, label in data_iterator:
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_iterator)
#因为随机初始化了模型,所以这个模型的精度大约为 1/num_outputs = 0.1

# e_accuracy = evaluate_accuracy(test_data,net)
# print(e_accuracy)


def SGD(params, learning_rate):
    for param in params:
        param[:] = param - learning_rate * param.grad

#训练
import sys
sys.path.append('..')

learning_rate = .1
epochs = 5
for i in range(epochs):
    train_loss = 0.
    train_acc = 0.

    for data, label in train_data:
        with ag.record():
            output = net(data)
            loss = cross_entropy(output, label)
        loss.backward()

        #将梯度做平均,这样学习率对 batch_size 不再敏感
        SGD(params, learning_rate / batch_size)
        train_loss = nd.mean(loss).asscalar()
        train_acc = accuracy(output, label)

    test_acc = evaluate_accuracy(test_data, net)
    print('Epoch %d. Train loss %f, Train accuracy %f, Test acc %f,' %
          (i, train_loss / len(train_data), train_acc / len(train_data), test_acc))
data, label = mnist_test[0:9]
show_image(data)
print('ture label')
print(get_text_labels(label))

predicted_labels = net(data).argmax(axis=1)
print('predict label')
print(get_text_labels(predicted_labels.asnumpy()))
