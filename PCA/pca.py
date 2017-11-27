# encoding:utf-8
from numpy import *

def loadDataSet(fileName, delim = '\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    dataArr = [map(float, line) for line in stringArr]
    return mat(dataArr)

def pca(dataMat, topNfeat = 9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigvalInd = argsort(eigVals)
    eigvalInd = eigvalInd[: -(topNfeat + 1): -1]    # 取最大的N个特征值对应的索引
    redEigVects = eigVects[:, eigvalInd]            # 权值矩阵
    lowDataMat = meanRemoved * redEigVects          # 降维后的数据
    reconMat = (lowDataMat * redEigVects.T) + meanVals  # 还原数据（有误差），PCA是有损压缩
    return lowDataMat, reconMat, redEigVects

def plotData(dataMat, reconMat):
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()

# 将NaN替换为平均值
def replaceNanWithMean():
    dataMat = loadDataSet('secom.data', ' ')
    numFeat = shape(dataMat)[1]
    for i in range(numFeat):
        meanVal = mean(dataMat[nonzero(-isnan(dataMat[:, i].A))[0], i])  # 计算所有非NaN的平均值
        dataMat[nonzero(isnan(dataMat[:, i].A))[0], i] = meanVal         # 将所有NaN值替换为平均值
    return dataMat

def main():
    dataMat = loadDataSet('testSet.txt')
    lowData, reconData, redEigVects = pca(dataMat, 1)
    # print lowData
    # print shape(reconData)
    # print reconData
    # print redEigVects * redEigVects.T
    # plotData(dataMat, reconData)

    ''' 利用PCA对半导体数据降维 '''
    secomData = replaceNanWithMean()
    pcaData, reData, weights = pca(secomData, 6)
    biasErr = secomData - reData    # 计算原始数据和还原数据的偏差
    varErr = var(biasErr, 0)        # 计算方差
    print reData
    print varErr
    print sqrt(sum(varErr))



if __name__ == '__main__':
    main()