#encoding:utf-8
from numpy import *
import spamFilter
import adbayes

#加载词表数据集和响应标签
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

#创建不重复词的列表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

#将文档转化为向量（词集模型）
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)        #创建一个所有元素均为0的向量
    for word in inputSet:
        if word in vocabList:               #文档中若出现了词汇表的单词，则文档向量的对应值设为1
            returnVec[vocabList.index(word)] = 1
        else:
            print "The word: %s is not in vocabulary!" % word
    return returnVec

#词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1   #对出现词频累加，而不是简单置1
    return returnVec

'''
@trainMatrix 文档向量的训练集
@trainCategory 每个文档向量的分类标签
@return p0Vect, p1Vect: p(w|c_i)每个类别中词条的条件概率（向量）
        pAbusive: p(c_i)每个类别的概率
'''
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # p0Num = zeros(numWords)
    # p1Num = zeros(numWords)
    # p0Denom = 0.0
    # p1Denom = 0.0
    # 解决概率乘积为0的情况，将所有词的出现次数初始化为1，分母初始化为2
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]             #向量累加计数
            p1Denom += sum(trainMatrix[i])      #词条总和计数
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)             #取对数相加，避免相乘的概率因子过小造成下溢出
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

'''
朴素贝叶斯分类器
@vec2Classify 待分类的向量
@p0Vec 分类为0的条件概率向量
@p1Vec 分类为1的条件概率向量
@pClass1 分类为1的概率
'''
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)       #向量的对应元素相乘（Numpy数组）
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

#测试贝叶斯分类器
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:                        #创建向量的训练集
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)     #训练得到条件概率和先验概率
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)  #分类待测向量
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)

def main():
    testingNB()
    spamFilter.spamFilter()
    adbayes.testWords()

if __name__ == '__main__':
    main()