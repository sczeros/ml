
#创建神经网络

from mxnet import ndarray as nd
from mxnet.gluon import nn

net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(256, activation = 'relu'))
    net.add(nn.Dense(10,activation ='sigmoid'))

print(net)

"""
Dense 普通的规则丶密集的全连接网络层
Block 完整的 可定制的网络定义
    nn.Block 是个一般化的构件,可当做一个神经网络,也可当做一层
    nn.Block 主要提供参数 描述forward如何自动执行 自动求导
Sequential 简易的网络定义
    nn.Sequential 是一个简单的nn.Block容器,通过add来添加nn.Block,
        动生成forward函数,把加进来的nn.Block逐一运行
"""

class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = nn.Dense(256)
            self.dense1 = nn.Dense(10)
    def forward(self, x):
        return self.dense1(nd.relu(self.dense0(x)))

net2 = MLP()
print(net2)
net2.initialize()
x = nd.random.uniform(shape=(4, 20))
print(x)
y = net2(x)
print(y)
print(nn.Dense)

print('default prefix:', net2.dense0.name)
net3 = MLP(prefix='another_mlp_')
print('customized prefix', net3.dense0.name)

class Sequential(nn.Block):
    def __init__(self, **kwargs):
        super(Sequential, self).__init__(**kwargs)

    def add(self, block):
        self._children.append(block)

    def forward(self, x):
        for block in self._children:
            x = block(x)
        return x
net4 = Sequential()
with net4.name_scope():
    net4.add(nn.Dense(256, activation='relu'))
    net4.add(nn.Dense(10))

net4.initialize()
y = net4(x)
print(y)

class ReclMLP(nn.Block):
    def __init__(self, **kwargs):
        super(ReclMLP ,self).__init__(**kwargs)
        self.net = nn.Sequential()
        with net.name_scope():
            self.net.add(nn.Dense(256, activation='relu'))
            self.net.add(nn.Dense(128, activation='relu'))
            self.dense = nn.Dense(64)
            #self.denses = [nn.Dense(256), nn.Dense(128), nn.Dense(64)]

        super()
    def forward(self, x):
        # x = self.net(x)
        # for dense in self.denses:
        #     x = dense(x)
        # return nd.relu(x)
        return nd.relu(self.dense(self.net(x)))

rec_mlp = nn.Sequential()
rec_mlp.add(ReclMLP())
rec_mlp.add(nn.Dense(10))
rec_mlp.initialize()
print(rec_mlp(x))
"""
Sequential(
  (0): RecMLP(
    (net): Sequential(
      (0): Dense(None -> 256, Activation(relu))
      (1): Dense(None -> 128, Activation(relu))
    )
    (dense): Dense(None -> 64, linear)
  )
  (1): Dense(None -> 10, linear)
)

Sequential(
  (0): ReclMLP(
    (net): Sequential(
      (0): Dense(None -> 256, Activation(relu))
      (1): Dense(None -> 128, Activation(relu))
    )
  )
  (1): Dense(None -> 10, linear)
)"""