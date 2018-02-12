#利用AdaBoost元算法提高分类性能
#当做重要决定时，大家可能都会考虑吸取多个专家而不只是一个人的意见。机器学习处理问
#题时有何尝不是如此？这就是元算法（meta-algorithm)背后的思路。

#third-party librares
import matplotlib.pyplot as plot
from numpy import *

def loadSimpData():
    datMat = matrix([[1. , 2.1],
                     [2. , 1.1],
                     [1.3 , 1.],
                     [1. , 1.],
                     [2. , 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

#基于单层决策树构建弱分类器
#单层决策树（decision stump,也称决策树桩)构建弱分类器。
#伪代码如下
#   将最小错误率minError设为+∞
#   对数据集中的每一个特征（第一层循环）：
#       对每个步长（第二层循环）：
#           对每个不等号（第三层循环）：
#               建立一棵单层决策树并利用加权数据集对它进行测试
#               如果错误率低于minError,则将当前单层决策树设为最佳单层决策树
#   返回最佳单层决策树

#通过阈值比较对数据进行分类的。所有在阈值一遍的数据会分到类别-1，而在另一边的数据分到类别+1.该函数可以
# 通过数组过滤来实现，首先将返回数据的全部元素设置为1，然后将所有不满足等式要求的元素设置为-1。可以基于
# 数据集中的任一元素进行比较，同时也可以将不等号在大于、小于之间切换。
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))
    #print(dataMatrix[:,dimen])
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

#遍历strumpClassify()函数所有的可能输入值，并找到数据集上最佳的单层决策树。这里“最佳”是
# 基于数据的权重向量D来定义的
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0#用于在特征的所有可能值上进行遍历
    bestStump = {}#存储给定权重D时所得到的最佳单层决策树的相关信息
    bestClasEst = mat(zeros((m,1)))
    minError = inf#初始化为无穷大,之后用于寻找可能的最小错误率
    #三层嵌套的for循环是程序最主要的部分。
    #第一层for循环在数据集的所有特征上遍历。考虑到数值型的特征，可以
    # 通过计算最小值和最大值来了解应该需要多大的步长
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        #第二层循环再在这些值上遍历。甚至将阈值设置为整个取值范围之外也是可以的。因此，
        #在取值范围之外还应该有两个额外的步骤。
        for j in range(-1,int(numSteps) + 1):# 0 到 numSteps之间并且包含
            #最后一个for循环则是在大于和小于之间切换不等式。
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                #如果predictedVals中的值不等于labelMat中真正的类别标签值，那么errArr的相应位置
                #为1.将错误向量errArr和权重D的相应元素相乘并求和，就得到了数值weightedError
                errArr = mat(ones((m,1)))#列向量
                errArr[predictedVals == labelMat] =0
                weightedError = D.T * errArr#这就是AdaBoost和分类器交互的地方。
                #基于权重D而不是其他错误计算指标来评价分类器的。如果需要使用其他分类器的话，就需要考虑D上
                #最佳分类器所定义的计算过程
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error "
                      "is %.3f" % (i,threshVal,inequal,weightedError))
                #将当前错误率与已有的最小错误率进行比较，如果当前的值较小，那么就在词典bestStump中保存该单层决策树。
                #字典，错误率和类别估计值都是返回给AdaBoost算法。
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst


"""
    对每次迭代：
        利用buildStump() 函数找到最佳的单层决策树
        将最佳单层决策树加入到单层决策树组
        计算alpha
        计算新的权重向量D
        更新累计类别估计值
        如果错误率等于0.0， 则退出循环
"""

#输入包括数据集 类别标签 迭代次数numIt, 其中numIt是在整个AdaBoost算法中唯一需要用户指定的参数
#
def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error, classEst = buildStump(dataArr,classLabels,D)
        print("D:" , D.T)
        alpha = float(0.5*log((1.0 - error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst: ",classEst.T)
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        print("aggClassEst: ",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) !=  mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print("total error: ",errorRate,"\n")
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst

# datMat, classLabels = loadSimpData()
# D = mat(ones((5,1))/5)
# buildStump(datMat,classLabels,D)
# classifierArr = adaBoostTrainDS(datMat,classLabels)


#AdaBoost分类函数
def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)

#
# adaClassify([0,0],classifierArr)

#自适应数据加载函数
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))

    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

datArr,labelArr = loadDataSet('horseColicTraining2.txt')
classifierArray, aggClassEst= adaBoostTrainDS(datArr , labelArr, 10)
#分类性能度量指标： 正确率、召回率及ROC曲线

#正确率 precision 等于 TP/(TP+FP),给出的是预测为正例的样本中真正正例的比例
#召回率 recall 等于 TP/(TP+FN) ,给出的是预测为正例的真实正例占所有真实正例的比例

#度量分类中的非均衡性的工具 ROC曲线 ROC curve ,ROC代表接收者操作特征(receiver operating characteristic)

#ROC曲线的绘制及AUC计算函数



def plotROC(predStrengths, classLables):
    cur = (1.0,1.0)
    ySum = 0.0
    numPosClas = sum(array(classLables) == 1.0)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLables) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plot.figure()
    fig.clf()
    ax = plot.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLables[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0],cur[0] - delX], [cur[1],cur[1] - delY], c='b')
        cur = (cur[0] - delX,cur[1] - delY)
    ax.plot([0,1],[0,1],'b--')
    plot.xlabel('False Positive Rate')
    plot.ylabel('True Positive Rate')
    plot.title('ROC curve for AdaBoost Horse Colic Detection Sytem')
    ax.axis([0,1,0,1])
    plot.show()
    print("the Area Under the Curve is: ", ySum*xStep)


plotROC(aggClassEst.T, labelArr)

