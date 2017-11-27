#encoding:utf-8
import bayes
from numpy import *

#文本解析
def textParse(String):
    import re
    listOfTokens = re.split(r'\W*', String)     #正则表达式,分割字符为非单词、数字字符（W大写）
    #切分并筛选出字符数多于2个的单词，转换为小写字符
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamFilter():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):              #导入并解析文本文件
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = bayes.createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):                 #留存交叉验证法随机构建训练集和测试集
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClass = []
    for docIndex in trainingSet:
        trainMat.append(bayes.setOfWords2Vec(vocabList, docList[docIndex]))     #获取向量的训练集
        trainClass.append(classList[docIndex])
    p0V, p1V, pSpam = bayes.trainNB0(array(trainMat), array(trainClass))        #训练数据得到条件概率和先验概率
    errorCount = 0          #分类错误计数
    for docIndex in testSet:
        wordVector = bayes.setOfWords2Vec(vocabList, docList[docIndex])         #对测试集分类
        if bayes.classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print 'The error rate is: ', float(errorCount) / len(testSet)