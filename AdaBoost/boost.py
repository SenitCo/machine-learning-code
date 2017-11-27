#encoding:utf-8
from numpy import *

def loadSimpData():
    dataMat = matrix([[1.0, 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

# 单层决策树生成函数
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

'''
@dataArr 输入样本集，@classLabels 标签集，@D 样本权值
@return bestStump 单层决策树，minError 最小误差，bestClassEst 分类标签
'''
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = mat(zeros((m, 1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                # print "split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f" % \
                #     (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst

'''
基于单层决策树的AdaBoost训练过程
@dataArr 样本集， @classLabels 标签集， @numIt 迭代次数
@return weakClassArr 弱训练器集， @aggClassEst 分类的标签的加权和
'''
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        #print "D:", D.T
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))     # 弱分类器权值，准确率高的分类器alpha权值大
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #print "classEst: ", classEst.T
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)     # 正确分类的样本的权重降低，错分样本的权重升高
        D = multiply(D, exp(expon))
        D = D / D.sum()        #样本权值，错分数据的权值大
        aggClassEst += alpha * classEst        # 分类标签的加权和
        #print "aggClassEst: ", aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print "total error: ", errorRate
        if errorRate == 0.0:    break
    return weakClassArr, aggClassEst

'''
AdaBoost分类函数
@dataToClass 一个或多个待分类样本
@classifierArr 多个弱分类器组成的数组
@return sign(aggClassEst) 分类标签
'''
def adaClassify(dataToClass, classifierArr):
    dataMatrix = mat(dataToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],   # 利用多个单层决策树分类并加权求和
                classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        #print aggClassEst
    return sign(aggClassEst)

# 自适应加载数据
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
    return dataMat, labelMat

'''
ROC曲线的绘制及AUC计算函数
@predStrengths (=aggClassEst)分类器的预测强度，预测分类标签的加权和
@classLabels 真实分类标签
'''
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClass = sum(array(classLabels) == 1.0)    #标签值为1的个数
    yStep = 1 / float(numPosClass)
    xStep = 1 / float(len(classLabels) - numPosClass)
    sortedIndices = predStrengths.argsort()     # 按值从小到大排列得到索引值
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndices.tolist()[0]:
        if classLabels[index] == 1.0:
            deltaX = 0
            deltaY = yStep
        else:
            deltaX = xStep
            deltaY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - deltaX], [cur[1], cur[1] - deltaY], c='b')
        cur = (cur[0] - deltaX, cur[1] - deltaY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis =([0, 1, 0, 1])
    plt.show()
    print "The area under the ROC curve is: ", ySum * xStep

def main():
    dataMat, classLabels = loadSimpData()
    m = shape(dataMat)[0]
    D = mat(ones((m, 1)) / m)
    BS = buildStump(dataMat, classLabels, D)
    print BS
    print m
    classifierArray, aggClassEst = adaBoostTrainDS(dataMat, classLabels, 9)
    print classifierArray
    ada = adaClassify([[0, 0], [5, 5]], classifierArray)
    print ada

    trainData, trainLabel = loadDataSet('horseColicTraining2.txt')
    adaClassifier, adaClassEst = adaBoostTrainDS(trainData, trainLabel, 10)
    testData, testLabel = loadDataSet('horseColicTest2.txt')
    adaPrediction = adaClassify(testData, adaClassifier)
    m = shape(testData)[0]
    errorArr = mat(ones((m, 1)))
    errorNum = errorArr[adaPrediction != mat(testLabel).T].sum()
    errorRate = errorNum / m
    print "errorNum = ", errorNum, "errorRate = ", errorRate
    plotROC(adaClassEst.T, trainLabel)


if __name__ == "__main__":
    main()