#自动求导

#在机器学习中，我们通常使用梯度下降来更新模型参数

from mxnet import ndarray as nd
from mxnet import autograd as ag

#假设我们想对函数 f=2×x2 求关于 x 的导数。我们先创建变量x，并赋初值。

x = nd.array([[1,2],[3,4]])
x.attach_grad()

with ag.record():
    y = x * 2
    z = y * x
z.backward()
print('x.grad : ', x.grad)
print(x)

def f(a):
    b = a * 2
    while nd.norm(b).asscalar() < 1000:
        b = b * 2
    if nd.sum(b).asscalar() > 0 :
        c = b
    else:
        c = 100 * b
    return c

with ag.record():
    y = x * 2
    z = y * x

head_gradient = nd.array([[10,1.],[.1,.01]])
z.backward(head_gradient)
print(z)
