from math import log
#运算符模块
import operator
import matplotlib.pyplot as plt

#数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet,labels
#获取西瓜数据集
def getDataSet(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    dataSize = len(arrayLines)
    labelList = arrayLines[0].strip().split(',')
    print(labelList)
    featureCount = len(labelList) - 3
    featureList = []#特征集
    index = 0
    for line in arrayLines:
        line = line.strip().split(',')
        feature = line[0:featureCount]
        feature.extend(line[-1])
        featureList.append(feature)
        index += 1
    #删除文件头
    del featureList[0]
    del labelList[-1]
    del labelList[-1]
    del labelList[-1]
    return featureList, labelList

# features,labels = getDataSet('西瓜数据集 3.0.csv')
# print(labels)

#计算给定数据集的香农熵
def calcShannoEnt(dataSet):
    numEntries = len(dataSet) #样本数量
    labelCounts = {}#样本标记
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannoEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannoEnt -= prob * log(prob, 2)
    return shannoEnt

#划分数据集
#dataSet 被划分的数据集 axis 划分数据集的特征 value 特征值
def splitDataSet(dataSet, axis, value):
    returnDataSet = [] #创建返回的数据集
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] #索引到axis停止
            reducedFeatVec.extend(featVec[axis+1:]) #从 axis+1又开始
            returnDataSet.append(reducedFeatVec) #reducedFeatVec一条记录，已经剔除value特征
    return returnDataSet

#选择划分属性
def choseBestFeatureToSplit(dataSet):
    numberFeatures = len(dataSet[0]) - 1 #特征个数

    baseEntropy = calcShannoEnt(dataSet)#计算当前数据集的信息熵
    baseInformationGain = 0.0
    baseFeature = -1
    for feature in range(numberFeatures) :
        featureList = [example[feature] for example in dataSet]#特征值
        featureValues = set(featureList)#特征可取值
        newEntropy = 0.0
        for value in featureValues:#计算每种划分方式的信息熵
            subDataSet = splitDataSet(dataSet, feature, value)#根据当前特征划分出的子集
            probability = len(subDataSet)/float(len(dataSet))
            newEntropy += probability * calcShannoEnt(subDataSet) #划分featureValues.length的子集数后总计的信息熵
        
        informationGain = baseEntropy - newEntropy #当前属性的信息增益
        
        if informationGain > baseInformationGain:  #选择更大的那个信息增益
            baseInformationGain = informationGain
            baseFeature = feature
    
    return baseFeature
def majoriryCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]
#创建树
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]#类别列表
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majoriryCnt(classList)
    bestFeature = choseBestFeatureToSplit(dataSet)
    bestFeaturelabel = labels[bestFeature]
    mytree = {bestFeaturelabel:{}}
    del labels[bestFeature]
    featValues = [featValue[bestFeature] for featValue in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        subLabels = labels[:]
        subDataSet = splitDataSet(dataSet,bestFeature,value)
        mytree[bestFeaturelabel][value] = createTree(subDataSet,subLabels)
    return mytree

myDataSet, labels = createDataSet()

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

# print(features)
#
# print(mytree)
tree = {'texture': {'清晰': {'root': {'蜷缩': '是', '硬挺': '否', '稍蜷':
    {'color': {'乌黑': {'touch': {'软粘': {'umbilical':
                                           {'是': '是', '否': '否'}}, '硬滑': '是'}}, '青绿': '是'}}}},
                    '模糊': '否', '稍糊': '否'}}

#在Python中使用matplotlib 注解绘制树形图
#Matplotlib提供了一个非常有用的注解工具annotations,它可以在数据图形上添加文本注解。注解通常用于解释数据的内容

#定义文本框和箭头形式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

#绘制带箭头的注解
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.axl.annotate(nodeTxt, xy=parentPt, xycoords = 'axes fraction', xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType,arrowprops=arrow_args)

# def createPlot():
#     fig = plt.figure(1, facecoujlor='white')
#     fig.clf()
#     createPlot.ax1 = plt.subplot(111, frameon=False)
#     plotNode('decionNode', (0.5, 0.1), (0.1, 0.5), decisionNode)
#     plotNode('leafNode', (0.8,0.1), (0.3,0.8), leafNode)
#     plt.show()

# createPlot()

#获取叶节点的数目
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

#获取树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.axl.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.axl = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

mytree = retrieveTree(0)

createPlot(mytree)

