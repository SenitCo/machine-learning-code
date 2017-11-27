#encoding:utf-8
import feedparser
import operator
import bayes
import spamFilter
from numpy import *

'''
使用朴素贝叶斯分类器从个人广告中获取区域倾向
'''
def calcMostFreq(vocabList, fullText):
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)         #统计词频
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:50]

def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = spamFilter.textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList = spamFilter.textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = bayes.createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])

    trainingSet = range(2 * minLen)
    testSet = []
    for i in range(5):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []
    trainClass = []
    for docIndex in trainingSet:
        trainMat.append(bayes.bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    print classList, len(classList)
    p0V, p1V, pSpam = bayes.trainNB0(array(trainMat), array(trainClass))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bayes.bagOfWords2VecMN(vocabList, docList[docIndex])
        if bayes.classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print 'The error rate is: ', float(errorCount) / len(testSet)
    return vocabList, p0V, p1V

def testWords():
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    vocabList, pSF, pNY = localWords(ny, sf)

    topNY = []
    topSF = []
    print pSF, '\n\n', pNY
    for i in range(len(pSF)):       #显示相关用词
        if -5.0 < pSF[i] < -4.0:
            topSF.append((vocabList[i], pSF[i]))
        if -5.0 < pNY[i] < -4.0:
            topNY.append((vocabList[i], pNY[i]))

    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]

    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]
