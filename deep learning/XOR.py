
#English version
#2017/12/13 night
#created by sczero
#implement artificial neural networks(ANN) to solve the XOR problem in Python

#中文版本
#2017/12/13 晚上
#我很忙 创建的这个文件
#实现人工神经网络解决异或问题



import numpy as np

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def s_prime(z):
    return np.multiply(sigmoid(z), sigmoid(1.0 - z))


#Epsilon是希腊语第五个字母艾普西隆的小写，写作ϵ或ε，常用于数学参数等的命名。
#layers 每层网络神经元的个数
def init_weights(layers, epsilon):
    weights = []
    for i in range(len(layers) - 1):
        w = np.random.rand(layers[i+1],layers[i]+1)
        w = w* 2*epsilon - epsilon
        weights.append(np.mat(w))
    return weights

def fit(X, Y, w, predict=False, x=None):
    w_grad = ([np.mat(np.zeros(np.shape(w[i]))) for i in range(len(w))])
    for i in range(len(X)):
        x = x if predict else X[0]
        y = Y[0,i]
        # forward propagate
        a = x
        a_s = []
        for j in range(len(w)):
            a = np.mat(np.append(1, a)).T
            a_s.append(a)
            z = w[j]*a
            a = sigmoid(z)
        if predict: return a
        #back propagate
        delta = a - y.T
        w_grad[-1] += delta*a_s[-1].T
        for j in reversed(range(1,len(w))):
            delta = np.multiply(w[j].T*delta, s_prime(a_s[j]))
            w_grad[j-1] +=(delta[1:] * a_s[j-1].T)
        return [w_grad[i]/len(X) for i in range(w)]

def predict(x):
    return fit(X, Y, w, True,x)

def test():
    # Notes
    # -----
    # Equivalent to x1` * `x2` in terms of array broadcasting.
    #
    # Examples
    # --------
    # >> > np.multiply(2.0, 4.0)
    # 8.0
    #
    # >> > x1 = np.arange(9.0).reshape((3, 3))
    # >> > x2 = np.arange(3.0)
    # >> > np.multiply(x1, x2)
    # array([[0., 1., 4.],
    #        [0., 4., 10.],
    #        [0., 7., 16.]])
    mat = np.multiply(2.0, 4.0)
    # [[0.  1.  2.]
    #  [3.  4.  5.]
    # [6.   7.  8.]]
    # [0.  1.  2.]
    # [[0.   1.   4.]
    #  [0.   4.  10.]
    # [0.    7.  16.]]
    x1 = np.arange(9.0).reshape((3, 3))
    x2 = np.arange(3.0)
    mat1 = np.multiply(x1, x2)
    print(x1)
    print(x2)
    print(mat1)

test()

