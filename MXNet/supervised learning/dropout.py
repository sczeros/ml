
#丢弃法

#还是来应对过拟合的问题
#训练数据小于测试数据 测试误差远远大于训练误差

from mxnet import ndarray as nd
from mxnet import autograd as ag
from mxnet import gluon
import mxnet as mx
import matplotlib.pyplot as plt

def dropout(x, drop_probability):
    keep_probability = 1 - drop_probability
    assert 0 <= keep_probability <= 1
    #这种情况下,把全部元素都丢弃
    if keep_probability == 0:
        return x.zero_like()
    #随机选择一部分该层的输出作为丢弃元素
    mask = nd.random.uniform(0, 1.0, shape = x.shape, ctx = x.context)\
           < keep_probability
    # 保证 E[dropout(X)] == X
    scale = 1 / keep_probability
    return mask * x * scale

A = nd.arange(20).reshape((5, 4))
print(A)
print(dropout(A, 0.5))