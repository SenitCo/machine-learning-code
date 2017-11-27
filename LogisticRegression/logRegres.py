#encoding:utf-8
from math import *
from numpy import *

#加载数据集
def loadDataset():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def Sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

#梯度上升算法（更新回归系数时需要遍历整个数据集）
def GradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)                 #转换为numpy矩阵数据类型
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))      #numpy矩阵？与ones(n)区分
    for k in range(maxCycles):
        h = Sigmoid(dataMatrix * weights)           #矩阵相乘
        error = (labelMat - h)
        weights += alpha * dataMatrix.transpose() * error
    return weights

#随机梯度上升算法（每次仅用一个样本点更新回归系数）
def stocGradAscent0(dataMatrix, classLabel):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = Sigmoid(sum(dataMatrix[i] * weights))
        error = classLabel[i] - h
        weights += alpha * error * dataMatrix[i]
    return weights

#改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabel, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01        #每次迭代时调整alpha
            randIndex = int(random.uniform(0, len(dataIndex)))  #随机选取样本更新
            h = Sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabel[randIndex] - h
            weights += alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataset()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

#Test：logistic regression分类函数
def classifyVector(inX, weights):
    prob = Sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabel = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabel.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabel, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = float(errorCount) / numTestVec
    print "The error rate of this test is: %f" % errorRate
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "After %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests))

def main():
    dataArr, labelMat = loadDataset()
    # weights1 = GradAscent(dataArr, labelMat)
    # plotBestFit(weights1)        #weights.getA()???
    # weights2 = stocGradAscent1(array(dataArr), labelMat)
    # plotBestFit(weights2)
    multiTest()

if __name__ == '__main__':
    main()