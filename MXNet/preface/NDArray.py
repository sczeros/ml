#MXnet 存储和变化数据的主要工具
#首先是CPU和GPU的异步计算，其次是自动求导

from mxnet import ndarray as nd
# array = nd.zeros((1,3))
# array1 = nd.ones((3,1))
# print(array)
# print(array1)

# array = nd.full((3,2), 4)
# print(array)

# array = nd.random_normal(0,1,shape=(3,4))
# print(array)
# print(array.T)

# x = nd.arange(0,9).reshape((3,3))
# print(x)
# print(x[2:2])
# x[1,2] = 9.0
# print(x)
# print(x[1:3,1:3])

# a = nd.arange(3).reshape((3,1))
# b = nd.arange(2).reshape((1,2))
# print('a : ' , a)
# print('b : ' , b)
# print('a + b : ', a + b)
import numpy as np

x = np.ones((2,3))
y = nd.array(x) # numpy to ndarray
z = y.asnumpy() # ndarray to numpy
print([z, y])