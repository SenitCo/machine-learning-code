#encoding:utf-8
from numpy import *
#加载数据集
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
    while(j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

'''
简化版SMO函数
参考文档：http://www.cnblogs.com/jerrylead/archive/2011/03/18/1988419.html
@dataMatIn 样本数据集
@classLabels 标签集
@C 常数
@toler 容错率
@maxIter 最大迭代次数
@return alpha, b
'''
def simpleSMO(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while(iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            #如果alpha可以更改进入优化过程
            if((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)       #随机选择第二个alpha
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if(labelMat[i] != labelMat[j]):         #保证alpha在0与C之间
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:          # C = 0 ?
                    print "L == H"
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T \
                    - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print "eta >= 0"
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print "j not moving enough"
                    continue
                #对i进行修改，修改量与j相同，但方向相反
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T \
                    - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T \
                    - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if(0 < alphas[i]) and (alphas[j] < C):
                    b = b1
                elif(alphas[j] > 0) and (alphas[j] > C):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print "iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged)
        if(alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
            print "iteration number: %d" % iter
    return b, alphas

'''
完整版Platt SMO的支持函数
'''
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))   #误差缓存

# 计算预测值与真实值的误差
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek


# 内循环中的启发式方法,用于选择第2个alpha值
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]     #返回矩阵基于的数组，非零值对应的索引
    if(len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
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

# 内部循环
def innerL(i, oS):
    Ei = calcEk(oS, i)
    # 判断每一个alpha是否被优化过，如果误差很大，就对该alpha值进行优化，toler是容错率
    if((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
            ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  # 使用启发式方法选取第2个alpha，选取使得误差最大的alpha
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        # 保证alpha在0与C之间
        if(oS.labelMat[i] != oS.labelMat[j]):       #当y1和y2异号，计算alpha的取值范围
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:                                       #当y1和y2同号，计算alpha的取值范围
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print "L == H"
            return 0
        ''' eta是alpha[j]的最优修改量，eta=K11+K22-2*K12,也是f(x)的二阶导数，K表示内积 '''
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        ''' 如果二阶导数-eta <= 0，说明一阶导数没有最小值，就不做任何改变，本次循环结束直接运行下一次for循环 '''
        if eta >= 0:
            print "eta >= 0"
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta    #利用公式更新alpha[j]，alpha2new=alpha2-yj(Ei-Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)        #判断alpha的范围是否在0和C之间
        updateEk(oS, j)      #在alpha改变的时候更新Ecache
        # 如果alphas[j]没有调整，就忽略下面语句，本次循环结束直接运行下一次for循环
        if(abs(oS.alphas[j] - alphaJold) < 0.00001):
            print "j not moving enough"
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        # 根据模型的公式计算b
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T \
            - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaJold) * oS.X[i, :] * oS.X[j, :].T \
            - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        ''' 根据公式确定偏移量b，理论上可选取任意支持向量来求解，但是现实任务中通常使用所有支持向量求解的平均值更加鲁棒 '''
        if(0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif(0 <oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1    # 如果有任意一对alpha发生改变，返回1
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
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
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

#计算权值参数w
def calcWeights(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w



