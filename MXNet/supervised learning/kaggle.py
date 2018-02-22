
#实战kaggle竞赛

#房价预测为例

#处理离散数据
#处理丢失的数据特征
#对数据进行标准化

#Get your hands dirty

import pandas as pd
import numpy as np

from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd as ag


#准备数据
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

all_x = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                     test.loc[:, 'MSSubClass':'SaleCondition']))


numeric_feats = all_x.dtypes[all_x.dtypes != 'object'].index
# Index(['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
#        'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
#        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
#        'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
#        'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
#        'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
#        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
#        'MoSold', 'YrSold'],
#       dtype='object')
all_x[numeric_feats] = all_x[numeric_feats].apply(
    lambda x : (x - x.mean()) / (x.std()))

all_x = pd.get_dummies(all_x, dummy_na = True)
all_x = all_x.fillna(all_x.mean())


num_train = train.shape[0]

x_train = all_x[ : num_train].as_matrix()
x_test = all_x[num_train : ].as_matrix()
y_train = train.SalePrice.as_matrix()

x_train = nd.array(x_train)
x_test = nd.array(x_test)
y_train = nd.array(y_train)
y_train.reshape((num_train, 1))

square_loss = gluon.loss.L2Loss()

def get_rmes_log(net, x_train, y_train):
    num_train = x_train.shape[0]
    clipped_preds = nd.clip(net(x_train), 1, float('inf'))
    return np.sqrt(2 *  nd.sum(square_loss(nd.log(clipped_preds),
                                           nd.log(y_train))).asscalar() / num_train)

def get_net():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(1))
    net.initialize()
    return net

import matplotlib as mpt
mpt.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt


def train(net, x_train, y_train, x_test, y_test, epochs, verbose_epochs,
          learning_rate, weight_decay):
    train_loss = []
    batch_size = 100
    if x_test is not None:
        test_loss = []
    data_set = gluon.data.ArrayDataset(x_train, y_train)
    data_iter = gluon.data.DataLoader(data_set, batch_size = batch_size, shuffle = True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate' : learning_rate,
                             'wd' : weight_decay})
    net.collect_params().initialize(force_reinit = True)
    for epoch in range(epochs):
        for data, label in data_iter:
            with ag.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)

            cur_train_loss = get_rmes_log(net, x_train, y_train)
        if epoch > verbose_epochs :
            print('Epoch %d, train loss : %f' % (epoch, cur_train_loss))
        train_loss.append(cur_train_loss)
        if x_test is not None:
            cur_test_loss = get_rmes_log(net, x_test, y_test)
            test_loss.append(cur_train_loss)

    plt.plot(train_loss)
    plt.legend(['train'])
    if x_test is not None:
        plt.plot(test_loss)
        plt.legend(['train', 'test'])
    plt.show()
    if x_test is not None:
        return cur_train_loss, cur_test_loss
    else:
        return cur_train_loss

#k折交叉认证

def k_fold_cross_validation1(k, epochs, verbose_epoch, X_train, y_train,
                       learning_rate, weight_decay):
    assert k > 1
    fold_size = X_train.shape[0] // k
    train_loss_sum = 0.0
    test_loss_sum = 0.0
    for test_i in range(k):
        X_val_test = X_train[test_i * fold_size: (test_i + 1) * fold_size, :]
        y_val_test = y_train[test_i * fold_size: (test_i + 1) * fold_size]

        val_train_defined = False
        for i in range(k):
            if i != test_i:
                X_cur_fold = X_train[i * fold_size: (i + 1) * fold_size, :]
                y_cur_fold = y_train[i * fold_size: (i + 1) * fold_size]
                if not val_train_defined:
                    X_val_train = X_cur_fold
                    y_val_train = y_cur_fold
                    val_train_defined = True
                else:
                    X_val_train = nd.concat(X_val_train, X_cur_fold, dim=0)
                    y_val_train = nd.concat(y_val_train, y_cur_fold, dim=0)
        net = get_net()
        train_loss, test_loss = train(
            net, X_val_train, y_val_train, X_val_test, y_val_test,
            epochs, verbose_epoch, learning_rate, weight_decay)
        train_loss_sum += train_loss
        print("Test loss: %f" % test_loss)
        test_loss_sum += test_loss
    return train_loss_sum / k, test_loss_sum / k

def k_fold_cross_validation(k, epochs, verbose_epoch, x_train, y_train,
                            learning_rate, weight_decay):
    assert k > 1
    fold_size = x_train.shape[0] // k
    train_loss_sum = 0.0
    test_loss_sum = 0.0
    for test_i in range(k):
        x_val_test = x_train[test_i * fold_size : (test_i + 1) * fold_size, :]
        y_val_test = y_train[test_i * fold_size : (test_i + 1) * fold_size]

        val_train_defined = False
        for i in range(k):
            if i != test_i:
                x_cul_fold = x_train[i * fold_size : (i + 1) * fold_size, :]
                y_cul_fold = y_train[i * fold_size : (i + 1) * fold_size]
                if not val_train_defined:
                    x_val_train = x_cul_fold
                    y_val_train = y_cul_fold
                    val_train_defined = True
                else:
                    x_val_train = nd.concat(x_val_train, x_cul_fold, dim = 0)
                    y_val_train = nd.concat(y_val_train, y_cul_fold, dim = 0)
        net = get_net()
        train_loss, test_loss = train(net, x_val_train, y_val_train,
                                      x_val_test, y_val_test, epochs, verbose_epoch,
                                      learning_rate, weight_decay)
        train_loss_sum += train_loss
        print('Test loss = %f' % test_loss)
        test_loss_sum += test_loss
    return train_loss_sum / k , test_loss_sum /k
k = 5
epochs  = 100
verbose_epoch = 95
learning_rate = 5
weight_decay = 0.0

train_loss, test_loss = k_fold_cross_validation(k, epochs, verbose_epoch,
                                                x_train, y_train, learning_rate,
                                                weight_decay)

print('%d-fold validation : Avg train loss: %f, Avg test loss : %f' %
      (k, train_loss, test_loss))