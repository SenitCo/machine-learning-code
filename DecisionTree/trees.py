#encoding:utf-8
#创建决策树分类器

from math import log
import operator
import treePlotter
#计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}                    #创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]      #取每一个数据样本的最后一个元素（类别）
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  #对所有类别计数
    shannonEnt = 0.0
    for key in labelCounts:             #计算概率和信息熵
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

#创建数据集
def createDataSet():
    dateSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dateSet, labels

'''
按照给定特征划分数据集
@dataSet 数据集
@axis 划分数据集的特征（索引）
@value （某一维）特征的取值
@return 符合特征取值的数据子集
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []             #函数中传递的是列表的引用，声明一个新列表对象
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[: axis]
            reducedFeatVec.extend(featVec[axis + 1 :])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1           #特征维数
    baseEntropy = calcShannonEnt(dataSet)       #整个数据集的原始熵
    bestInfoGain = 0.0                          #最大的信息增益
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]      #列表推导式，取样本中的某一维特征值
        uniqueVals = set(featList)              #创建唯一的分类标签列表（set()无序不重复的元素集）
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)    #对每一维特征划分数据子集并计算信息熵
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

'''
如果数据集处理完所有属性，类标签依然不唯一，则采用多数表决的方法决定叶子结点的分类
@classList 类标号列表
@return 票数最多的分类
'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

'''
@dataSet 数据集
@labels 所有特征的标签
@return 创建的树
'''
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):     #类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:                                #遍历完所有特征时返回出现次数最多的类别
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}                #创建树的字典
    del(labels[bestFeat])                       #删除相应特征维
    featValues = [example[bestFeat] for example in dataSet]    #得到列表包含的所有属性值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]                                   #递归创建决策树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

'''
决策树分类器
@inputTree 创建的决策树
@featLabels 属性特征标签
@testVec 待测试的向量
@return 分类标签（结果）
'''
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)      #将标签字符喘转换为索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

#使用pickle模块存储决策树
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

def main():
    # #利用海洋生物数据判断是否属于鱼类
    # myData, labels = createDataSet()
    # labelsPara = labels[:]
    # print myData
    # print labels
    # print chooseBestFeatureToSplit(myData)
    # myTree = createTree(myData, labelsPara)
    # print myTree
    # storeTree(myTree, 'DecisionTree_Fish.txt')
    # storeTr = grabTree('DecisionTree_Fish.txt')
    # print storeTr
    # print classify(storeTr, labels, [0, 1])
    # treePlotter.createPlot(storeTr)

    #利用隐形眼镜数据集预测隐形眼镜类型
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    print lenses
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    print lensesTree
    storeTree(lensesTree, 'DecisionTree_Lenses.txt')
    treePlotter.createPlot(lensesTree)

if __name__ == '__main__':
    main()

