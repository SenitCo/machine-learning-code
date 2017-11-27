#encoding:utf-8
from numpy import *

''' 分类回归树(CART) '''

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)   # 将每行数据映射为浮点数
        dataMat.append(fltLine)
    return dataMat

'''
二分数据集
@dataSet 待划分数据集 @feature 待切分的特征 @value 特征值
@return 数据划分后的两个子集
'''
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

# 建立叶节点的函数（求均值）
def regLeaf(dataSet):
    return mean(dataSet[:, -1])

# 误差计算函数（总体方差）
def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]

'''
树构建函数
@dataSet 数据集 @leafType 建立叶节点的函数
@errType 误差计算函数 @ops 树构建所需其他参数的元组
'''
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

# 回归树切分函数
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]   # 容许的误差下降值
    tolN = ops[1]   # 切分的最小样本数
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:     # 如果所有值相等则退出
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):         # matrix类型不能被hash
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if(shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue

''' 回归树后剪枝函数 '''
# 测试输入变量是否是一棵树
def isTree(obj):
    return (type(obj).__name__ == 'dict')

# 递归遍历树，找到两个叶节点并计算平均值
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

'''
后剪枝函数：合并降低误差就将叶节点合并
@tree 待剪枝的树 @testData 剪枝所需的测试数据
'''
def prune(tree, testData):
    if shape(testData)[0] == 0:     # 没有测试数据则对树进行塌陷处理
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, 1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print "merging"
            return treeMean
        else:
            return tree
    else:
        return tree

''' 模型树(Model Tree),叶节点设定为分段线性函数 '''
def linearSolve(dataSet):
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]   # 将X和Y中数据格式化，X第一列为1
    Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse, tring increasing the second value o fops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

# 模型树建立叶节点的函数（线性分段）
def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws

# 模型树误差计算函数
def modelError(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))

def plotData(dataMat):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, -2].flatten().A[0], dataMat[:, -1].flatten().A[0])
    plt.show()

# 回归树预测
def regTreeEval(model, inData):
    return float(model)

# 模型树预测
def modelTreeEval(model, inData):
    n = shape(inData)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1:n+1] = inData
    return float(X * model)

# 树回归进行预测
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat

def main():
    # dataList = loadDataSet('ex2.txt')
    # dataMat = mat(dataList)
    # plotData(dataMat)
    # treeDict = createTree(dataMat, ops=(0, 1))  # 创建回归树
    # print treeDict
    #
    # testData = loadDataSet('ex2test.txt')
    # testData = mat(testData)
    # prune(treeDict, testData)               # 后剪枝

    # modelData = loadDataSet('exp2.txt')     # 创建模型树
    # modelData = mat(modelData)
    # modelTree = createTree(modelData, modelLeaf, modelError, (1, 10))
    # print modelTree

    trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    m, n = shape(testMat)
    RegTree = createTree(trainMat, regLeaf, regErr, ops=(1, 20))
    ModelTree = createTree(trainMat, modelLeaf, modelError, ops=(1, 20))
    yReg = createForeCast(RegTree, testMat[:, 0:n-1], regTreeEval)            # 回归树预测
    yModel = createForeCast(ModelTree, testMat[:, 0:n-1], modelTreeEval)      # 模型树预测
    print corrcoef(yReg, testMat[:, -1], rowvar=0)[0, 1]    # 预测值和真实值的相关系数（越大预测效果越好）
    print corrcoef(yModel, testMat[:, -1], rowvar=0)[0, 1]
    standWS, X, Y = linearSolve(trainMat)
    yStdL = mat(zeros((m, 1)))
    for i in range(m):
        yStdL[i] = testMat[i, 0:n-1] * standWS[1:n] + standWS[0]
    print corrcoef(yStdL, testMat[:, -1], rowvar=0)[0, 1]    # 标准线性回归的相关系数

if __name__ == '__main__':
    main()










