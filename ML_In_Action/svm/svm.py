#支持向量机
from numpy import *
#SMO算法中的辅助函数
def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i, m):
    j = i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

dataArr, labelArr = loadDataSet("testSet.txt")

#简化版SMO算法
#dataMatIn 数据集 classLables 类别标签 常数C 容错率 toler 退出前最大的循环次数
#实例 dataArr, labelArr, 0.6, 0.001, 40
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    print(dataMatrix[0,:].T)
    # dataMatrix
    # [[3.54248500e+00   1.97739800e+00]
    #  ................................
    # [2.38700300e+00   1.57313100e+00]]
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaParisChanged = 0
        for i in range(m):
            #类别multiply(alphas,labelMat).T 矩阵的转置
            # [[-0. -0.  0. -0.  0.  0.  0. -0. -0. -0. -0. -0. -0.  0. -0.  0.  0. -0.
            # 0. -0. -0. -0.  0. -0. -0.  0.  0. -0. -0. -0. -0.  0.  0.  0.  0. -0.
            # 0. -0. -0.  0. -0. -0. -0. -0.  0.  0.  0.  0.  0. -0.  0.  0. -0. -0.
            # 0.  0. -0.  0. -0. -0. -0. -0.  0. -0.  0. -0. -0.  0.  0.  0. -0.  0.
            # 0. -0. -0.  0. -0.  0.  0.  0.  0.  0.  0.  0. -0. -0. -0. -0.  0. -0.
            # 0.  0.  0. -0. -0. -0. -0. -0. -0. -0.]]
            fxi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            #误差
            Ei = fxi - float(labelMat[i])
            #在if语句中，不管是正间隔还是负间隔都会被测试。并且在该语句中，也要同时检查alpha值，以保证其不能等于0或C。由于
            #后面alpha小于0或大于C时将被调整为0或C，所以一旦在该if语句中它们等于这两个值得话，那么它们就已经在“边界”上了
            #，因而不在能够减小或增大，因此也就不值得再对它们进行优化了。
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                #随机 选择第二个alpha值，即alpha[i]
                j = selectJrand(i,m)#得到的j必不等于i
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b  #计算类别的公式
                Ej = fXj - float(labelMat[j])#误差
                #保存选出来两个alphas I J old 的值
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                #保证alpha在0与C 之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                #eta是alpha[j]的最优修改量
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                #此处对smo算法进行了简化。如果eta为0，那么计算新得alpha[j]就麻烦了，这里不作详细介绍。现实中，这种情况不常发生，因此
                #忽略这一部分也无伤大雅。于是，可以计算出一个新的alpha[j],然后斗鱼L和H值对进行调整。
                if eta >= 0:
                    print("eta>=0")
                    continue
                #计算出一个新的alpha[j]
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                #检查alpha[j]是否有轻微改变。如果是的话，就退出for循环，然后，alpha[i]和alpha[j]同样进行改变，虽然改变的相同，但方向相反
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                #对i进行修改，修改量与j相同，但方向相反
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                #对alpha[i] 和 alpha[j]进行优化之后，给这两个alpha值设置一个常数项
                b1 = b - Ei - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j] - alphaJold)*\
                                                                                                      dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j] - alphaJold)*\
                                                                                                      dataMatrix[j,:]*dataMatrix[j,:].T

                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                # 最后，在优化过程结束的同时，必须确保在合适的时机结束循环。如果程序执行到for循环的最后一行都不执行continue语句，
                # 那么就已经成功的改变了一对alpha值是否做了更新，如果有更新则将iter设为0后继续执行程序，只有在所有数据集上遍历maxIter次
                #，且不再发生任何alpha修改之后，程序才会停止并退出while循环
                alphaParisChanged += 1
                print("iter: %d i:%d, paris changed %d" %(iter,i,alphaParisChanged))
        if (alphaParisChanged == 0):
            iter += 1
        else: iter = 0
        print("iteration number: %d" %iter)
    return b,alphas

# smoSimple(dataArr, labelArr, 0.6, 0.001, 40)

#完整版Platt SMOde 支持函数
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        #误差缓存
        self.eCache = mat(zeros((self.m,2)))
        # self.K = mat(zeros((self.m, self.m)))
        # for i in range(self.m):
        #     self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)

def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

#内循环中的启发式方法
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]
    validEchacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEchacheList)) > 1:
        for k in validEchacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):#选择具有最大步长的j
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

#完整Platt SMO算法中的优化历程
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i,oS,Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L == H")
            return 0
        eta = 2.0*oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j)#更新误差缓存
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS,i)
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*\
                                                                                           (oS.alphas[j] - alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*\
                                                                                           (oS.alphas[j] - alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 <oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0

#完整版Platt SMO 的外循环代码
def smoP(dataMatIn, classLables, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn),mat(classLables).transpose(), C,toler)
    iter = 0
    entireSet = True
    alphaPairChanged = 0
    while (iter < maxIter) and ((alphaPairChanged > 0) or (entireSet)):
        alphaPairChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairChanged))
                iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairChanged == 0):
            entireSet = True
        print("interation number: %d" % iter)
    return oS.b,oS.alphas

dataArr, labelArr = loadDataSet('testSet.txt')
b, alphas = smoP(dataArr,labelArr, 0.6, 0.001, 40)
def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i], X[i,:].T)
    return w

# ws = calcWs(alphas,dataArr,labelArr)
# print(ws)

#核转换函数 kTup 是一个包含核函数信息的元组，元组的第一个参数是描述所用核函数类型的一个字符串
#其他两个参数则都是核函数可能需要的可选参数
def kernelTrans(X, A, kTup):
    m, n =shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin':
        K = X * A.T
    elif kTup[0]=='rbf':
        #径向基函数的计算
        for j in range(m):
            deltaRow = X[j ,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('Houston We Have a Proble -- That Kernel is not recognized')
    return K

def smoP1(dataMatIn, classLables, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct1(mat(dataMatIn),mat(classLables).transpose(), C,toler,kTup)
    iter = 0
    entireSet = True
    alphaPairChanged = 0
    while (iter < maxIter) and ((alphaPairChanged > 0) or (entireSet)):
        alphaPairChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairChanged += innerL1(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairChanged))
                iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairChanged += innerL1(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairChanged == 0):
            entireSet = True
        print("interation number: %d" % iter)
    return oS.b,oS.alphas

class optStruct1:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        #误差缓存
        self.eCache = mat(zeros((self.m,2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)

def innerL1(i, oS):
    Ei = calcEk1(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i,oS,Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L == H")
            return 0
        #eta = 2.0*oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        eta = 2.0*oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j)#更新误差缓存
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS,i)
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.K[i,i] - oS.labelMat[j]*\
                                                                                           (oS.alphas[j] - alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.K[i,j] - oS.labelMat[j]*\
                                                                                           (oS.alphas[j] - alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 <oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0


def calcEk1(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP1(dataArr, labelArr, 200, 0.0001, 10000, ('rbf',k1))
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i,:],('rbf',k1))
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i,:],('rbf',k1))
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))

testRbf()

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat,hwLabels

def testDigits(kTup=('rbf',10)):
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP1(dataArr,labelArr, 200, 0.0001, 10000,kTup)
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i,:],kTup)
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the raining error rate is: %f" % (float(errorCount)/m))
    dataArr, labelArr = loadImages('testDigits')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i,:],kTup)
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))

testDigits(('rbf', 20))