#k-Nearest Neighbor , KNN
#k-近邻算法

#导入 numpy 类库（科学计算包）
from numpy import *
#与画数据图有关的库
import matplotlib
import matplotlib.pyplot as plt

#运算符模块
import operator

#基本通用的函数，简单数据集
def createDataSet():
    #array 简单理解 就是把列表的列表数据转化为矩阵形式，API看了一点，不能舍本逐末了
    #想要矩阵 调用 numpy.arrayy 接口就好
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

# group, labels = createDataSet();
#
# print(group.shape[0])
# print(labels)

#k-近邻算法  实现 KNN 分类算法
#     对未知类别属性的数据集中的每个点依次执行以下操作：
#      (1)计算已知类别数据集中的点与当前点之间的距离
#      (2)按照距离递增次序排序
#      (3)选取与当前点距离最小的K个点
#      (4)确定前K个点所在类别的出现频率
#      (5)返回前K个点出现频率最高的类别作为当前点的预测分类
#
#

#   classify0([0,0],group,labels,3)
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #矩阵的 shape 返回行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    #tile(a, (2, 2))
    # array([[0, 1, 2, 0, 1, 2],
    #      [0, 1, 2, 0, 1, 2]])
    #tile([0,0],(4,1))
    # [[0,0]
    #  [0,0]
    #  [0,0]
    #  [0,0]]
    sqDiffMat = diffMat**2#矩阵或列表中各个元素的平方
    sqDistances = sqDiffMat.sum(axis = 1) #每个元素相加求和
    distances = sqDistances**0.5 #开方 得到距离

    sortedDistIndicies = distances.argsort() #按距离从小到大排序结果序列的索引所组织的列表
    classCount = {}#存储类别的字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] #根据 i 索引值索引出 排序产生的 sortedDistIndicies[i] 存储距离升序的原索引值 列表
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #把类比添加到字典中去，如果是字典中没有的类别，就创建；有的话就++
    # 把某类别频率或者说概率大的放在字典的第一个位置上
    sortedDistIndicies = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedDistIndicies[0][0] #返回kNN算法的分类结果

#classfier = classify0([0,0],group,labels,3)
# print(classfier)

#将文本记录转换为 NumPy的解析程序
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberFromLines = len(arrayOLines)
    # np.zeros((2, 1)) numpy.zeros 函数
    # array([[0.],
    #        [0.]])
    returnMat = zeros((numberFromLines,3)) # 一个矩阵类型 元素值全为 0. 行数为 numberFromLines 列数为 3
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3] #特征矩阵
        #用文件中的数据覆盖原0.的数 每一个是一个样本 此变量只存样本属性信息，
        # 相对应的类别信息在classLabelVector
        classLabelVector.append(int(listFromLine[-1])) #类别信息 与returnMat 的行索引一一对应
        index += 1
    return returnMat,classLabelVector

# datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
# print(datingDataMat[0][1])
# print(datingLabels)
# print(datingDataMat[:,1])

#分析数据：使用Matplotlib 创建散点图
#散点图绘制过程
#fig = plt.figure() #得到坐标图的框架
# ax = fig.add_subplot(111)#调节图的倍数
# ax.scatter(datingDataMat[:,1], datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
#前两个参数分别是x y 的坐标值 后两个利用类别不同属性 在散点图中绘制了色彩不等、尺寸不同的点
#plt.show()#执行画图

#归一化数值 newValue = (oldValue - min) / (max - min) 将数值处理为0到1或者-1到1之间。
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet -tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet,ranges,minVals

# normat, ranges, minVals = autoNorm(datingDataMat)
# print(normat)

def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normat, ranges, minVals = autoNorm(datingDataMat)
    m = normat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classfierResult = classify0(normat[i,:],normat[numTestVecs:m,:],
                                    datingLabels[numTestVecs:m],3)
        print("the classfier came back with : %d, the real answer is: %d"
              %(classfierResult,datingLabels[i]))
        if(classfierResult != datingLabels[i]):
            errorCount += 1;
    print("the total error rate is; %f" %(errorCount/float(numTestVecs)))

datingClassTest()

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input(""))