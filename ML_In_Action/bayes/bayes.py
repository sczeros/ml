'''
2017 11 27 中午创建
由 sczero 创建
'''

from numpy import *
#朴素贝叶斯

#词表到向量的转换函数
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec


def createVocabList(dataSet):
    vocabSet = set([])#创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)#创建两个集合的并集
    return list(vocabSet)

#创建一个其中所有元素都为0的向量
#vocabList 单词列表 inputSet 需要与单词列表相对比的样本
#该函数的输入输出为词汇表及某个文档，输出的是文档向量，向量的每一个元素为1或0，分别表示词汇表中的单词在输入文档中是否出现。
#首先创建一个和词汇表等长的向量，并将其元素都设为0.遍历文档中的单词，如果出现了词汇表中的单词，则将输入文档向量中的对应值设为1.
#通俗的讲就是把输入文本转化成标准的向量和单词表一样长度
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word : %s is not in my Vocabulary! " %word)
    #返回值类型示例 [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    return returnVec

#局部测试
# listOPosts, listClasses = loadDataSet()
# myVocabSet = createVocabList(listOPosts)
# numDataSet = setOfWords2Vec(myVocabSet,listOPosts[0])
#
# trainMat = []
# for postinDoc in listOPosts:
#     trainMat.append(setOfWords2Vec(myVocabSet,postinDoc))

# print(array(trainMat))
# print(trainMat)
# print(myVocabSet)
# print(listClasses)


#朴素贝叶斯分类器训练函数
#trainMatrix
# [[1 1 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 1 0 0]
#  [0 1 0 0 1 0 0 0 0 1 0 0 1 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0]
#  [1 0 1 0 1 0 0 0 0 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1]
#  [0 0 0 0 0 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0]
#  [1 0 0 0 1 0 0 0 0 1 0 1 0 0 0 0 1 0 0 0 0 0 1 0 0 1 1 1 0 0 0 0]
#  [0 1 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0]]

#trainCategory
#[0, 1, 0, 1, 0, 1]
#该函数的伪代码

#计算每个类别中的文档书目
#对每篇训练文档：
#    对每个类别：
#        如果词条出现在此文档中-》增加该词条的计数值
#        增加所有词条的计数值
#对每个类别：
#    对每个词条：
#        将该词条的数据除以总词条数目得到的条件概率
#返回每个类别的条件概率
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # p0Num = zeros(numWords)
    # p1Num = zeros(numWords)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:#是含侮辱信息的文本
            p1Num += trainMatrix[i]
            #[ 0.  1.  0.  1.  1.  0.  0.  0.  1.  0.  0.  0.  0.  1.  2.  0.  1.  3.
            #0.  0.  0.  0.  0.  2.  1.  1.  0.  0.  1.  1.  1.  1.]
            p1Denom += sum(trainMatrix[i])
            #19.0
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p1Vect = p1Num/p1Denom
    # p0Vect = p0Num/p0Denom
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect,p1Vect,pAbusive

# p0V, p1V, pAb = trainNB0(trainMat, listClasses)
# print(p0V)
# print(p1V)
# print(pAb)

#朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #实践中常通过取对数的方式来将“连乘”转化为“连加”以避免数值下溢
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    print(p0)
    print(p1)
    if p1 > p0:
        return 1
    else:
        return 0

#测试函数
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabSet = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabSet,postinDoc))
    pOV, p1V, pAb = trainNB0(trainMat, listClasses)

    testEntry = ['love', 'garbage', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabSet, testEntry))
    print(testEntry, 'classified as : ',classifyNB(thisDoc,pOV,p1V,pAb))

    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabSet, testEntry))
    print(testEntry, 'classified as : ', classifyNB(thisDoc,pOV,p1V,pAb))

# testingNB()

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i ).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    pOV,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),pOV,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ',float(errorCount)/len(testSet))

#spamTest()