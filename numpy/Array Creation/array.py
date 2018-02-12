
import numpy as np
import matplotlib.pyplot as plt
#arange
#start stop step

# array3 = []
# array1 = np.arange(6)
# array1 = array1.reshape(2,3)
# array2 = np.arange(0,5,2,array3)
# print(array1)
# print(array2)
# print(array3)

#linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
#在start 和 stop 区间之间的数 区间生成 num 生成等差数列的数量 endpoint 默认为True，最后一个数为 stop
# retstep 默认为False 如果为True 返回间隔
# array1 = np.linspace(0,2,5,True,True)
# #[ 0.   0.4  0.8  1.2  1.6]
# print(array1)

# N = 8
# y = np.zeros(N)
# x1 = np.linspace(0, 10, N,endpoint=True)
# x2 = np.linspace(0, 10, N,endpoint=False)
# plt.plot(x1,y,'o')
# plt.plot(x2, y + 0.5 , 'o')
# plt.xlim([-0.5,12]) #[xmin,xmax]
# plt.ylim([-0.5, 1])#y坐标轴的最小值与最大值
# plt.show()

#logspace
#geomspace


#array
# array = np.array([[2, 1, 3]], ndmin=2,order='K')
# print(array)

#asarray

#numpy.empty(shape, dtype=float, order='C')


#Return a new array with the same shape and type as a given array. but it is empty without
# initializing entries
#array = np.empty([2,2],dtype=int)
# array = np.empty([2, 2], dtype=float)
# print(array)
#numpy.empty_like(a, dtype=None, order='K', subok=True)¶
# array = np.empty_like(([1,2,3], [4,5,6]))
# print(array)


#numpy.eye(N, M=None, k=0, dtype=<type 'float'>)
#生成一个对角线为1 其他元素为0的方阵
#diagonal 对角线
#Return a 2-D array with ones on the diagonal and zeros elsewhere.
# N : int
# Number of rows in the output.
#
# M : int, optional
# Number of columns in the output. If None, defaults to N.
#
# k : int, optional
# Index of the diagonal: 0 (the default) refers to the main diagonal, a
# positive value refers to an upper diagonal, and a negative value to a
# lower diagonal.
# array = np.eye(4, k =3)
# print(array)
#
# numpy.identity(n, dtype=None)[source]
# Return the identity array.
#
# The identity array is a square array with ones on the main diagonal.
# array = np.identity(3)
# print(array)

#numpy.ones(shape, dtype=None, order='C')[source]
# Return a new array of given shape and type, filled with ones.
# array = np.ones((2,3))
# print(array)
#zeros continue

# numpy.full(shape, fill_value, dtype=None, order='C')[source]
# Return a new array of given shape and type, filled with fill_value.

# array = np.full((2,3), fill_value=1254)
# print(array)

#
#print(np.pi)

#numpy.random.randn(d0, d1, ..., dn)
# Return a sample (or samples) from the “standard normal” distribution.
#
# If positive, int_like or int-convertible arguments are provided, randn
# generates an array of shape (d0, d1, ..., dn), filled with random floats
# sampled from a univariate “normal” (Gaussian) distribution of mean 0
# and variance 1 (if any of the d_i are floats, they are first converted
# to integers by truncation).
# A single float randomly sampled from the distribution is returned if
# no argument is provided.

# This is a convenience function. If you want an interface that takes a
# tuple as the first argument, use numpy.random.standard_normal instead.
# array = np.random.randn(2,3,4)
# print(array)

# array = np.random.rand(2,3)
# print(array)
# array = np.random.randint(9, size=(2,3))
# print(array)

# numpy.random.random_sample(size=None)
# Return random floats in the half-open interval [0.0, 1.0).
#
# Results are from the “continuous uniform” distribution over the stated
# interval. To sample Unif[a, b), b > a multiply the output of random_sample
# by (b-a) and add a:

# array = np.random.random_sample((2,3))
# print(array)

array = np.all([1,-2])
print(array)
array = np.all([[True,True],[True,True]])
print(array)
o = np.array([False])
z = np.all([-1, 4, 5], out=o)
# o = np.array([False])
# z = np.all([-1,4,5],out=o)
x = id(o)
y = id(z)
print(o)
print(z)
print(x)
print(y)