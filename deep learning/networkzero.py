
#2018 01 04
#周志华机器学习神经网络算法编程

#standard library
import random

#third liibraries
import numpy as np

#反向传播算法


class NetworkZero(object):
    def __init__(self,sizes):
        """
        sizes 列表 神经网络中每层神经元的个数
        len(sizes)表示这个神经网络有多少层
        各个连接权与阈值被初始化为随机变量，在各自不同的限定域中
        :param sizes:
        """
        self.num_layer = len(sizes)#神经网络的层数
        self.sizes = sizes#[d,q,l]
        # randn 第一行元素 输入层神经元到隐层第一个神经元的权重
        #       第二行元素 输入层神经元到隐层第二个神经元的权重
        #       。。。
        #       第n行元素 输入层神经元到隐层第n个神经元的权重
        self.weights_input_hidden = [np.random.randn(sizes[1],sizes[0])]#输入层到隐层之间的连接权
        self.weights_hidden_output = [np.random.randn(sizes[2],sizes[1])]#隐层到输出层之间的连接权
        self.threshold_hidden = [np.random.randn(sizes[1],1)]#隐层的阈值
        self.threshold_output =[np.random.randn(sizes[2],1)]#输出层的阈值

    def feedforward(self,a,weights,threshold):
        """
        前向输出

        输入a，根据输入确定输出结果
        """
        return sigmoid(np.dot(weights,a) + threshold)



<<<<<<< HEAD
    def backprop(self,x,y, eta = 0.1):
=======
    def backprop(self,x,y):
>>>>>>> 7547b1ac580471d8ccf23dbbefd2d7b944d34d67
        """
        输出是一个一个进行的
        :param x: 是个元组，里面有两个元素，第一个脐部，第二个是根蒂
        :param y: 对训练样例x的真实输出
        :return:
<<<<<<< HEAD
                   误差逆传播算法
                       输入：训练集 D = {}
                             学习率Nita
                       过程：
                       1：在（0，1）范围内随机初始化网络中所有连接权和阈值
                       2：repeat
                       3:  for all(x k, y k) ∈ D do
                       4:      根据当前参数式 5.3 计算当前样本的输出 y k
                       5:      根据式 5.10 计算输出层神经元的梯度项g j
                       6:      根据式 5.15 计算隐层神经元的梯度项 e h
                       7:      根据式 5.11 - 5.14 更新连接权 w hj, v ih 与阈值
                       8：  end for
                       9: until 达到停止条件
                       输出：连接权与阈值确定的多层前馈神经网络
=======
>>>>>>> 7547b1ac580471d8ccf23dbbefd2d7b944d34d67
        """
        #先进行输入处理
        #先处理隐层
        alphas = []#隐层的输入
        betas = []
        activation_hiddens = []#隐层的激活值
<<<<<<< HEAD

        activation = [x]
        for t,w in zip(self.threshold_hidden,self.weights_input_hidden):
            alpha = np.dot(w,activation)#第 h 个隐层神经元的输入，元组中的每一个输入元素
            alphas.append(alpha)
            activation_hidden = sigmoid(alphas - t)#第 h 个隐层神经元的激活，元组中的每一个激活元素
            activation_hiddens.append(activation_hidden)

        print(np.array(alphas))
        print(t)
        print(np.array(alphas) - np.array(t).T)
        #再处理输出层
        activation_outputs = []#输出层的激活值
        for w, t in zip(self.weights_hidden_output, self.threshold_output):
            beta = np.dot(w, activation_hiddens)  # 第 j 个输出神经元的输入
=======
        for w,t in zip(self.weights_input_hidden,self.threshold_hidden):
            alpha = np.dot(w,x)#第 h 个隐层神经元的输入，元组中的每一个输入元素
            alphas.append(alpha)
            activation_hidden = sigmoid(alpha - t)#第 h 个隐层神经元的激活，元组中的每一个激活元素
            activation_hiddens.append(activation_hidden)

        #再处理输出层
        activation_outputs = []#输出层的激活值
        for w, t in zip(self.weights_hidden_output, self.threshold_output):
            beta = np.dot(w, activation_hidden)  # 第 j 个输出神经元的输入
>>>>>>> 7547b1ac580471d8ccf23dbbefd2d7b944d34d67
            betas.append(beta)
            activation_output = sigmoid(beta - t)
            # activation_outputs.append(activation_output)
        # for activation_hidden in activation_hiddens:
        #     for w,t in zip(self.weights_hidden_output,self.threshold_output):
        #         beta = np.dot(w,activation_hidden)#第 j 个输出神经元的输入
        #         betas.append(beta)
        #         activation_output = sigmoid(beta - t)
        #         activation_outputs.append(activation_output)

<<<<<<< HEAD
        #前向传播处理完毕 。得到 activation_output 输出啦（计算出来的）

        #计算均方误差

        #输出层神经元的梯度项
        g = activation_output * (1 - activation_output) * (y - activation_output)
        w = eta * g * activation_hiddens
        #输出层神经元的阈值梯度
        threshold_output = -1 * eta * g

        #隐层神经元的梯度
        e = activation_hiddens * (1 - np.array(activation_hiddens).T) * self.weights_hidden_output * g
        v = eta * e * x
        #隐层神经元的阈值梯度
        threshold_hidden = -1 * eta * e

        #更新隐层与输出层之间的参数
        self.weights_input_hidden = [old + new for old, new in zip(self.weights_input_hidden,w)]
        self.threshold_output +=  threshold_output

        #更新输入层与隐层之间的参数
        self.weights_input_hidden = [old + new for old,new in zip(self.weights_input_hidden,v)]
        self.threshold_hidden += threshold_hidden

    def evaluate(self,x,y):
        epochs = 30#训练次数
        for j in range(epochs):
            for eachX,eachY in zip(x,y):
                self.backprop(eachX,eachY,0.1)

        print(self.weights_input_hidden)
        print(self.weights_hidden_output)
        print(self.threshold_hidden)
        print(self.threshold_output)
        print("start evaluate")

=======
        #前向传播处理完毕 。得到 activation_outputs 输出啦（计算出来的）

        #计算均方误差
        ave_err = activation_output - y

        #输出层神经元的梯度项
        g = activation_output * (1 - activation_output) *(activation_output - y)
        #隐层神经元的梯度
        e =
        """
            误差逆传播算法
                输入：训练集 D = {}
                      学习率Nita
                过程：
                1：在（0，1）范围内随机初始化网络中所有连接权和阈值
                2：repeat
                3:  for all(x k, y k) ∈ D do
                4:      根据当前参数式 5.3 计算当前样本的输出 y k
                5:      根据式 5.10 计算输出层神经元的梯度项g j
                6:      根据式 5.15 计算隐层神经元的梯度项 e h
                7:      根据式 5.11 - 5.14 更新连接权 w hj, v ih 与阈值
                8：  end for
                9: until 达到停止条件
                输出：连接权与阈值确定的多层前馈神经网络
            """
>>>>>>> 7547b1ac580471d8ccf23dbbefd2d7b944d34d67
    def error_function(self,activation_output,y):
        return activation_output - y


def sigmoid(inX):
    """激活函数"""
    return 1/(1 + np.exp(-inX))

def sigmoid_prime(inX):
    """激活函数的导数"""
    return sigmoid_prime(inX)*(1 - sigmoid_prime(inX))

<<<<<<< HEAD
def loadWatermelon():
    fs = open("../西瓜数据集 3.0.csv")
    arrayOLines = fs.readlines()
    arrayOLines.pop(0)
    numberLines = len(arrayOLines)
    data = np.zeros((numberLines,2))
    label = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split(',')
        data[index,:] = listFromLine[6:8]
        label.append(listFromLine[-1])
        index += 1
    labels = []
    for element in label:
        if element == '是':
            labels.append(1)
        else:
            labels.append(0)
    return data,labels

data,label = loadWatermelon()

network = NetworkZero([2,2,1])
network.evaluate(data,label)
=======
d = 2
q = 3
l = 1
network = NetworkZero([d,q,l])
>>>>>>> 7547b1ac580471d8ccf23dbbefd2d7b944d34d67
