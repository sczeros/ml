
#使用高层抽象包实现线性回归

from mxnet import ndarray as nd
from mxnet import autograd as ag
from mxnet import gluon

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(shape=y.shape)

batch_size = 10
dataset = gluon.data.ArrayDataset(X, y)
data_iter = gluon.data.DataLoader(dataset,batch_size = 10, shuffle= True)
for data,label in data_iter:
    print(data,label)
    break

#定义模型
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))

net.initialize()

square_loss = gluon.loss.L2Loss()
