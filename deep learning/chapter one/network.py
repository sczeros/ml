"""
network.py
~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#复杂的算法 ≤≤ 简单的学习算法 + 好的训练数据


####libraries
#standard library
import random

#Third-party library
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
           respective layers of the network.  For example, if the list
           was [2, 3, 1] then it would be a three-layer network, with the
           first layer containing 2 neurons, the second layer 3 neurons,
           and the third layer 1 neuron.  The biases and weights for the
           network are initialized randomly, using a Gaussian
           distribution with mean 0, and variance 1.  Note that the first
           layer is assumed to be an input layer, and by convention we
           won't set any biases for those neurons, since biases are only
           ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        """[array([[-0.63265541],
       [-1.18768536],
       [-0.18171312]]), array([[-1.22959297]])]"""
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]#偏移器，衡量感知器触发的难易程度
        """[array([[-0.17366775, -0.18752093],
       [-1.1832174 , -0.27367587],
       [ 0.749856  ,  1.74714423]]), array([[-0.74112412, -0.90667563,  0.34559348]])]"""
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

    def feedforward(self,a):
        """Return the output of network if "a"  is input"""
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a


    #随机梯度下降 stochastic gradient descent
    #随机梯度下降的核心是，梯度是期望。期望可使用小规 模的样本近似估计。具体而言，在算法的每一步
    #，从训练集中均匀抽出一小批mini_batch量样本。小批量的数目通常是一个相对较小的数，从一到一百。重要的是，
    #当训练集大小增长时，数目通常是固定的。可能在拟合几十亿的样本时，每次更新计算只用到几百个样本。
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
           gradient descent.  The ``training_data`` is a list of tuples
           ``(x, y)`` representing the training inputs and the desired
           outputs.  The other non-optional parameters are
           self-explanatory.  If ``test_data`` is provided then the
           network will
           be evaluated against the test data after each
           epoch, and partial progress printed out.  This is useful for
           tracking progress, but slows things down substantially."""

        # eta 学习的速率η
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        #for j in xrange(epochs):#迭代的次数
        for j in range(epochs):
            random.shuffle(training_data)#随机的打乱训练数据的顺序
            #将数据分为合适大小的mini_batch
            mini_batches = [training_data[k : k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            # 对每一个epochs,使用一次梯度下降算法。通过self.update_mini_batch(mini_batch,eta)实现，这段代码会利用
            # mini_batch中的训练数据，通过一个梯度下降的循环来更新神经网络的权重系数和偏移量。
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            #如果 test_data 被提供了，那么在每一个训练“代”结束之后都会评估神经网络的表现，然后输出部分进展，
            #这对跟踪进展非常有用，但会大大减慢速度。（所以在算法完成测试之后，这段代码可以注释掉？）
            if test_data:
                print("Epoch {0} : {1}/{2}".format(j,self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self,mini_batch,eta):
        """Update the network's weights and biases by applying
                gradient descent using backpropagation to a single mini batch.
                The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
                is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            #大部分工作都是这一行代码做的
            #逆传播 backpropagation的算法，快算计算代价函数Cost function的梯度.因此update_mini_batch所做的仅仅就是对
            #mini_batch中每一个训练数据样本计算这些梯度，然后更新self.weights 和 self.biases
            #delta 希腊语字母表第四字母δ
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            #向量微分算子，Nabla算子（nabla operator），
            nabla_b = [nb + dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw + dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        self.weights = [w - (eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)]

    #反向传播算法
    def backprop(self, x, y):
        """
            反向传播这个术语经常被误解为用于多层神经网络的整个学习算法。实际上，反向传播
        仅指用于计算梯度的方法，而另一种算法，例如随机梯度下降SGD,使用该梯度来进行学习。此外
        ，反向传播经常被误解为仅适用于多层神经网络，但是原则上它可以计算任何函数的导数（对于
        一些函数，正确的响应是报告函数的导数是未定义的）。

        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #feedforward
        activation = x
        activations = [x] #list to store all the activations ,layer by layer
        zs = [] #list to store all the z vector, layer by layer
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backword pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b,nabla_w)

    def evaluate(self,test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    #损失函数 loss function 或 代价函数 cost function 或 误差函数 error function
    def cost_derivative(self,output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)




#### Miscellaneous functions
def sigmoid(inX):
    """The sigmoid function"""
    return 1.0 / (1.0 + np.exp(-inX))

#用来计算 sigmoid 函数的倒数
def sigmoid_prime(inX):
    """Deviation of this sigmoid function"""
    return sigmoid(inX)*(1 - sigmoid(inX))

import mnist_loader

training_data,validatoin_data, test_data = mnist_loader.load_data_wrapper()

network = Network([784,30,10])
network.SGD(training_data, 30, 10, 3.0, test_data=test_data)

def test():
    # x = [1, 2, 3]
    #
    # y = [4, 5, 6]
    #
    # z = [7, 8, 9]
    #
    # for i,j,k in zip(x[0:-1],y[0:-1],z[0:-1]):
    #     print(i)
    #     print(j)
    #     print(k)
    # sizes = [2, 3, 1]
    # print(sizes[-1])
    # biases = [np.random.randn(y,1) for y in sizes[1:]]
    # print(biases)
    # for x, y in zip(sizes[:-1], sizes[1:]):
    #     print(x , y)
    # weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]
    # print(weights[1])
    # print(weights)
    mini_batch_size = 10
    n = 50
    # for k in range(0, n, mini_batch_size):
    #     print(k)
test()