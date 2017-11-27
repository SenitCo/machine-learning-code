#encoding:utf-8

from numpy import *
import matplotlib.pyplot as plt

# 加载数据集
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

# 标准回归函数
def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:      # 计算矩阵行列式
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * yMat)    # 计算权值向量
    return ws

# 绘制散点图和拟合直线
def plotData(xArr, yArr, ws):
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat1 = xMat * ws
    print corrcoef(yHat1.T, yMat)  # 计算相关系数

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()

# 局部加权线性回归函数
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

# 测试原始数据集
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

# 绘制散点图和LWLR的拟合曲线
def plotLWLR(xArr, yArr, yHat):
    xMat = mat(xArr)
    srtInd = xMat[:, 1].argsort(0)      # 矩阵类型，按值排序得到索引
    xSort = xMat[srtInd][:, 0, :]       # xMat[srtInd]为三维矩阵，所以要切片操作
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')     # 散点图
    ax.plot(xSort[:, 1], yHat[srtInd])      # 拟合直线
    plt.show()

# 回归预测误差计算函数
def rssError(yArr, yHat):
    return ((yArr - yHat) ** 2).sum()

# 预测鲍鱼数据集的年龄
def predictAbaloneAge():
    abX, abY = loadDataSet('abalone.txt')
    # 在原始数据上的测试
    yHat01 = lwlrTest(abX[0: 99], abX[0: 99], abY[0: 99], 0.1)
    yHat1 = lwlrTest(abX[0: 99], abX[0: 99], abY[0: 99], 1)
    yHat10 = lwlrTest(abX[0: 99], abX[0: 99], abY[0: 99], 10)
    err01 = rssError(abY[0: 99], yHat01.T)
    err1 = rssError(abY[0: 99], yHat1.T)
    err10 = rssError(abY[0: 99], yHat10.T)
    print err01, err1, err10
    # 在新数据集上的测试
    yHatNew01 = lwlrTest(abX[100: 199], abX[0: 99], abY[0: 99], 0.1)
    yHatNew1 = lwlrTest(abX[100: 199], abX[0: 99], abY[0: 99], 1)
    yHatNew10 = lwlrTest(abX[100: 199], abX[0: 99], abY[0: 99], 10)
    errNew01 = rssError(abY[100: 199], yHatNew01.T)
    errNew1 = rssError(abY[100: 199], yHatNew1.T)
    errNew10 = rssError(abY[100: 199], yHatNew10.T)
    print errNew01, errNew1, errNew10

# 岭回归
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

# 标准化
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)
    inVar = var(inMat, 0)
    inMat = (inMat - inMeans) / inVar
    return inMat

# 前向逐步回归（Iasso的简易实现）
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
            # ws = wsMax.copy()    # 累积改变所有属性的权重
        ws = wsMax.copy()   # 每次迭代时改变误差降低最快的属性权值
        returnMat[i, :] = ws.T  # 记录每次迭代的权重
    return returnMat

'''预测LEGO价格'''
# 购物信息的获取函数
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    from time import sleep
    import json
    import urllib2
    sleep(10)
    myAPIstr = 'AIzaSyDsvOXMUNyk96jiq3W6kfsVzTxfrDxS6Mk'
    searchURL = 'http://www.googleeapis.com/shopping/search/v1/public/prducts?key=%scountry=US&q=lego+%d&alt=json' \
        (myAPIstr, setNum)      # URL 404 错误
    pg = urllib2.urlopen(searchURL)
    retDict = json.load(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print "%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice)
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print 'Problem with item %d' % i

def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 42.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

def getLegoData():
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    m, n = shape(lgX)
    lgX1 = mat(ones((m, n + 1)))
    lgX1[:, 1: n + 1] = mat(lgX)    # 将lgX copy 到lgX1,且第一列元素置为常数1
    return lgX, lgY

# 交叉验证测试岭回归
def crossValidation(xArr, yArr, numVal=10):
    m = len(yArr)
    numLam = 30     # 30个不同的lambda值
    indexList = range(m)
    errorMat = zeros((numVal, numLam))
    for i in range(numVal):
        trainX = []     # 创建训练集和测试集容器
        trainY = []
        testX = []
        testY = []
        random.shuffle(indexList)   # 对列表元素混洗，实现数据集和测试集数据点的随机选取
        for j in range(m):
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)        # wMat为numLam x n矩阵（n为数据维数）
        for k in range(numLam):
            matTestX = mat(testX)
            matTrainX = mat(trainX)
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain    # 用训练数据的参数将测试数据标准化
            yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)  # 预测估计（对应前面对真实Y值进行了中心化）
            errorMat[i, k] = rssError(yEst.T.A, array(testY))
    meanErrors = mean(errorMat, 0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)]    # 从30个lambda对应的权值向量中选出误差最小的
    xMat = mat(xArr)
    yMat = mat(yArr).T
    meanX = mean(xMat, 0)
    varX = var(xMat, 0)
    unReg = bestWeights / varX          # 数据(权值)还原
    const = -1 * sum(multiply(meanX, unReg)) + mean(yMat)    # y = wx + b
    print 'bestWeights: \n', bestWeights
    print "The best model from Ridge Regression is: \n ", unReg
    print "with constant term: ", const

def testAbalone():
    abX, abY = loadDataSet('abalone.txt')
    m, n = shape(abX)
    abX1 = mat(ones((m, n + 1)))
    abX1[:, 1 : n + 1] = mat(abX)
    standWS = standRegres(abX, abY)
    standWS1 = standRegres(abX1, abY)
    print standWS.T
    print standWS1.T
    crossValidation(abX, abY, 10)   # 权值和standWS1更接近




def main():
    # '''**********************************'''
    # xArr, yArr = loadDataSet('ex0.txt')
    # yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    # plotLWLR(xArr, yArr, yHat)
    # predictAbaloneAge()
    # '''**********************************'''
    # abX, abY = loadDataSet('abalone.txt')
    # ridgeWeights = ridgeTest(abX, abY)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax1.plot(ridgeWeights)
    # '''**********************************'''
    # IterWeights = stageWise(abX, abY, 0.001, 5000)
    # print IterWeights
    # ax2 = fig.add_subplot(212)
    # ax2.plot(IterWeights)
    # plt.show()
    # '''**********************************'''
    testAbalone()

if __name__ == "__main__":
    main()