#监督学习基础

from mxnet import ndarray as nd
from mxnet import autograd as ag
from matplotlib import pyplot as plt

#linear regression
#随机数值 X[i]  标记数值Y[i]
#y[i] = 2 * X[i][0] - 3.4 * X[i][1] + 4.2 + noise

num_layer = 2
num_example = 1000

true_weight = [2,-3.4]
true_biase = 4.2
X = nd.random_normal(shape=(num_example,num_layer))
Y = true_weight[0] * X[:,0] + true_weight[1] * X[:,1] + true_biase
Y += .01 * nd.random_normal(shape= Y.shape)
# print(X[0],Y[0])
#
# plot.scatter(X[:,1].asnumpy(), Y.asnumpy())
# plot.show()


import random
batch_size = 10
def data_iter():
    #产生一个随机索引
    idx = list(range(num_example))#[0,1,2,……，999]
    random.shuffle(idx)#打乱idx
    for i in range(0,num_example,batch_size):
        j = nd.array(idx[i:min(i + batch_size,num_example)])
        yield nd.take(X,j), nd.take(Y,j)

# for data,label in data_iter():
#     print(data,label)
#     break

#随机初始化模型参数
w = nd.random_normal(shape=(num_layer,1))
b = nd.zeros((1,1))
params = [w,b]

#创建梯度
for param in params:
    param.attach_grad()

#定义模型
def net(X):
    return nd.dot(X,w) + b

#损失函数
def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2

#优化
def SGD(params, lr):
    for param in params:
        param[:] = param[:] - lr * param.grad

#模型函数
def real_fn(X):
    return 2 * X[:,0]  - 3.4 * X[:, 1] + 4.2



#绘制损失函数与训练次数降低的折线图，以及预测值和真实值之间的散点图
def plot(losses, X, sample_size = 1000):
    xs = list(range(len(losses)))
    f, (fg1, fg2) = plt.subplots(1, 2)
    # f = plt.figure()
    # fg1 = f.add_subplot(121)
    # fg2 = f.add_subplot(122)
    fg1.set_title('Loss during training')
    fg1.plot(xs, losses, '-r')
    fg2.set_title('Estimate vs real function')
    fg2.plot(X[:sample_size, 1].asnumpy(), net(X[:sample_size, :]).asnumpy(), 'or', label = 'Estimated')
    fg2.plot(X[:sample_size, 1].asnumpy(), real_fn(X[:sample_size, :]).asnumpy(), '*g', label = 'Real')
    #fg2.legend()
    plt.show()

epochs = 5
learning_rate = 0.001
niter = 0
losses = []
moving_losses = 0
smothing_constant = 0.01
#训练
for e in range(epochs):
    total_loss = 0

    for data,label in data_iter():
        with ag.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        SGD(params,learning_rate)
        total_loss += nd.sum(loss).asscalar()

        #记录每读取一个记录后，损失的移动平均值的变化
        niter += 1
        curr_loss = nd.mean(loss).asscalar()
        moving_losses = (1 - smothing_constant) * moving_losses  + (smothing_constant) * curr_loss

        #correct the bias from the moving  averages
        est_loss = moving_losses / (1 - (1 - smothing_constant) ** niter)

        if (niter + 1) % 100 == 0:
            losses.append(est_loss)
            print("Epoch %s, batch %s. Moving avg of loss: %s."
                  "Average loss : %f" % (e, niter, est_loss, total_loss / num_example))
            plot(losses,X)
