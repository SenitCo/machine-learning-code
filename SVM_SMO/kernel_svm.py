#encoding:utf-8
from numpy import *
import svm

'''
以下是有核函数的版本
http://www.cnblogs.com/tonglin0325/p/6107114.html
'''

'''
核转换函数
@X 样本矩阵
@A 某一个样本
@kTup 核函数
@return 变换后的样本（高维空间）
'''
def kernelTrans(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':    # 线性核函数
        K = X * A.T
    elif kTup[0] == 'rbf':  # RBF径向基函数
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1]**2))
    else:
        raise NameError('That kernel is not recognized')
    return K

class optStructK:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))  # first column is valid flag
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)

def calcEkK(oS, k):  # 计算误差
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJrandK(i, m):
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j

def clipAlphaK(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

# 内循环中的启发式方法,用于选择第2个alpha值
def selectJK(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]     #返回矩阵基于的数组，非零值对应的索引
    if(len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEkK(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrandK(i, oS.m)
        Ej = calcEkK(oS, j)
        return j, Ej

def updateEkK(oS, k):
    Ek = calcEkK(oS, k)
    oS.eCache[k] = [1, Ek]

# 内部循环的代码和简版的SMO代码很相似
def innerL(i, oS):
    Ei = calcEkK(oS, i)
    # 判断每一个alpha是否被优化过，如果误差很大，就对该alpha值进行优化，toler是容错率
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
        (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJK(i, oS, Ei)  # 使用启发式方法选取第2个alpha，选取使得误差最大的alpha
        alphaIold = oS.alphas[i].copy();
        alphaJold = oS.alphas[j].copy();
        # 保证alpha在0与C之间
        if (oS.labelMat[i] != oS.labelMat[j]):  # 当y1和y2异号，计算alpha的取值范围
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:  # 当y1和y2同号，计算alpha的取值范围
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H: print "L==H"; return 0
        # eta是alpha[j]的最优修改量，eta=K11+K22-2*K12,也是f(x)的二阶导数，K表示核函数
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  # changed for kernel
        # 如果二阶导数-eta <= 0，说明一阶导数没有最小值，就不做任何改变，本次循环结束直接运行下一次for循环
        if eta >= 0: print "eta>=0"; return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta  # 利用公式更新alpha[j]，alpha2new=alpha2-yj(Ei-Ej)/eta
        oS.alphas[j] = clipAlphaK(oS.alphas[j], H, L)  # 判断alpha的范围是否在0和C之间
        updateEkK(oS, j)  # 在alpha改变的时候更新Ecache
        print "j=", j
        print oS.alphas.A[j]
        # 如果alphas[j]没有调整，就忽略下面语句，本次循环结束直接运行下一次for循环
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])  # update i by the same amount as j
        updateEkK(oS, i)  # 在alpha改变的时候更新Ecache
        print "i=", i
        print oS.alphas.A[i]
        # 已经计算出了alpha，接下来根据模型的公式计算b
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
        oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
        oS.alphas[j] - alphaJold) * oS.K[j, j]
        # 根据公式确定偏移量b，理论上可选取任意支持向量来求解，但是现实任务中通常使用所有支持向量求解的平均值，这样更加鲁棒
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1  # 如果有任意一对alpha发生改变，返回1
    else:
        return 0


'''
Platt SMO的外循环
@dataMatIn 样本数据集
@classLabels 样本分类标签
@C 常参数
@toler 容错率
@maxIter 最大迭代次数
@kTup 核函数
@return 参数 b, alpha
'''
def PlattSMO(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStructK(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    # 有alpha改变同时遍历次数小于最大次数，或者需要遍历整个集合
    while(iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        # 首先进行完整遍历，过程和简化版的SMO一样
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print "fullSet, iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged)
            iter += 1
        else:       #非边界遍历，挑选其中alpha值在0和C之间非边界alpha进行优化
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print "non-bound, iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged)
            iter += 1
        if entireSet:    #如果这次是完整遍历的话，下次不用进行完整遍历
            entireSet = False
        elif(alphaPairsChanged == 0):   #如果alpha的改变数量为0的话，再次遍历所有的集合一次
            entireSet = True
        print "iteration number: %d" % iter
    return oS.b, oS.alphas

def testRbf(k1=1.3):
    dataArr, labelArr = svm.loadDataSet('testSetRBF.txt')
    b, alphas = PlattSMO(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))  # C=200 important
    datMat = mat(dataArr);
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]    # 取非零alphas值的索引，构建支持向量的矩阵
    sVs = datMat[svInd]  # get matrix of only support vectors
    labelSV = labelMat[svInd]
    print "there are %d Support Vectors" % shape(sVs)[0]
    m, n = shape(datMat)
    errorCount = 0
    '''
    w = Sigma(alpha_i * y_i * phi(x_i)),<Sigma为求和>
    g_svm(x)=sign(w.T*phi(x)+b)=sign(Sigma(alpha_i*y_i*phi(x_i)*phi(x))+b)=sign(Sigma(alpha_i*y_i*kernel(x_i,x))+b)
    <kernel()为核函数，phi()为高维空间的映射函数，alpha_i>0,对应的x_i为支持向量>
    '''
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount) / m)
    dataArr, labelArr = svm.loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat = mat(dataArr);
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the test error rate is: %f" % (float(errorCount) / m)



def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))    #目录名/文件名
    return trainingMat, hwLabels

def testDigits(kTup=('rbf', 10)):
    dataArr, labelArr = loadImages('digits/trainingDigits')
    b, alphas = PlattSMO(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = mat(dataArr);
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]     # 训练得到支持向量
    labelSV = labelMat[svInd];
    print "there are %d Support Vectors" % shape(sVs)[0]
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)   # 利用核函数和支持向量转换样本
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b   # 预测分类
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount) / m)
    dataArr, labelArr = loadImages('digits/testDigits')
    errorCount = 0
    datMat = mat(dataArr);
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)   # sVs是训练集的支持向量，dataMat是测试集数据
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the test error rate is: %f" % (float(errorCount) / m)

def main():
    #testRbf()
    testDigits()

if __name__ == '__main__':
    main()