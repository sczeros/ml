#西瓜数据集 bayes分类器

from numpy import *

#获取西瓜数据集
def getDataSet(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    dataSize = len(arrayLines)
    featureCount = len(arrayLines[0].split(',')) - 1
    featureList = []#特征集
    labelList = []#标签集
    index = 0
    for line in arrayLines:
        line = line.strip().split(',')
        featureList.append(line[0:featureCount])
        labelList.append(line[-1])
        index += 1
    #删除文件头
    del featureList[0]
    del labelList[0]
    return featureList, labelList

features,labels = getDataSet('西瓜数据集 3.0.csv')

#
