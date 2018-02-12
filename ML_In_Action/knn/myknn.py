

#徒手写机器学习算法系列之机器学习十大算法 k-NN（k near neighbor)k近邻算法
#计算方法 在n维空间中求亮点之间的距离而已

from numpy import *

#具体的分类方法
#将新数据中的特征与样本集中的特征进行比较，然后算法提取样本集中特征最相似数据（最近邻）的分类标签
#inX 没有标签的新数据 dataSet“训练”数据集 labels分类类型 k
def classifies(xDataSet, dataSet, labels, k):
    # 1.准备数据，矩阵化
    # 数据行数 样本数量
    m = dataSet.shape[0]
    difference = tile(xDataSet, (m, 1)) - dataSet
    differenceSquare = difference**2
    differenceSum = differenceSquare.sum(axis=1)
    distances = differenceSum**0.5
    sortedDistancesIndex = distances.argsort()
    classCount = {}
    for i in range(k):
        classLabelsValue = labels[sortedDistancesIndex[i]]
        classCount[classLabelsValue] = classCount.get(classLabelsValue,0) + 1

    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


def getDataSetFromFile(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    numbersFormLines = len(arrayLines)

    returnMat = zeros((numbersFormLines, 3))
    classLabelsVector = []
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat = listFromLine[0:3]
        classLabelsVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelsVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = normDataSet.shape[0]
    normDataSet = normDataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


