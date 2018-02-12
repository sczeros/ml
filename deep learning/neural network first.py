#感知机和多层网络
from numpy import *
#实例： 学习 XOR
#XOR函数（“异或”逻辑）是两个二进制值x1和x2的运算
#数据集[[0,0],[0,1],[1,0],[1,1]] [0,1,1,0]

def loadDataSet():
    dataSet = [[0,0],[0,1],[1,0],[1,1]]
    lables = [0,1,1,0]
    return dataSet, lables
#Sigmoid函数
def sigmoid(inX):
    return 1/(1 + exp(-inX))

#梯度下降法
def gradientDescent(dataSet,labels):
    dataMatrix = mat(dataSet)
    labelsMatrix = mat(labels).transpose()
    m, n = shape(dataMatrix)
    #学习率
    alpha = 0.5
    #训练次数
    maxCycle = 500
    weights = ones((m,1))
    for k in range(maxCycle):
        #单个神经元训练权值
        print('梯度下降法')

#感知机学习模型
def perceptronModel(dataSet,labels):
    dataMatrix = mat(dataSet)