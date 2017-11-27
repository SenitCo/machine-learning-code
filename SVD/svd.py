#encoding:utf-8
from numpy import *

def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]

''' 相似度计算 '''
def euclidSim(inA, inB):    # 欧式距离
    return 1.0 / (1.0 + linalg.norm(inA - inB))

def pearsSim(inA, inB):     # 皮尔逊相关系数
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]   # [-1, 1]归一化到[0, 1]

def cosSim(inA, inB):       # 余弦距离
    num = float(inA.T * inB)
    denom = linalg.norm(inA) * linalg.norm(inB)
    return 0.5 + 0.5 * (num / denom)

'''
基于物品相似度的评分系统
@dataMat 数据集 @user 用户 @simMeas 相似度计算方式 @item 未评分项
'''
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0: continue
        overLap = nonzero(logical_and(dataMat[:, item] > 0, dataMat[:, j] > 0))[0]  # 确定两个均被评级的物品的共同用户索引
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])   # overLap为索引(列向量)数组
        print 'overLap-->', overLap
        # print 'The %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating  # 利用相似度计算加权评分
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal

''' 基于协同过滤的推荐引擎 '''
def recommend(dataMat, user, N = 3, simMeas = cosSim, estMethod = standEst):
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]  # 未评级物品的索引(行向量)数组
    # print 'unratedItems-->', unratedItems
    if len(unratedItems) == 0:
        return 'You rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]   # 寻找前N个未评级物品

def changeData(dataMat):
    dataMat[0, 1] = dataMat[0, 0] = dataMat[1, 0] = dataMat[2, 0] = 4
    dataMat[3, 3] = 2

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

# 基于SVD的评分估计
def svdEst(dataMat, user, simMeas, item,):
    numEig = 4
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = linalg.svd(dataMat)
    Sig = mat(eye(numEig) * Sigma[:numEig])
    xformedItems = dataMat.T * U[:, :4] * Sig.I
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:    continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        # print "The %d and %d similarity is: %f" % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:   return 0
    else:   return ratSimTotal / simTotal

# 基于SVD的图像压缩
def printMat(inMat, thresh = 0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print 1,
            else:
                print 0,
        print ' '

def imgCompress(numSV = 3, thresh = 0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print "************original matrix*************"
    printMat(myMat, thresh)
    U, Sigma, VT = linalg.svd(myMat)
    # sumSig = sum(Sigma ** 2)
    # cntSig = 0
    # for i in range(len(Sigma)):     # 取总能量占比达到90%的奇异值数量
    #     cntSig += Sigma[i]
    #     if float(cntSig) / sumSig > 0.9:
    #         break
    # numSV = i
    # print numSV, cntSig
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print "reconstructed matrix using %d singular values" % numSV
    printMat(reconMat, thresh)

def main():
    data = loadExData()
    U, Sigma, VT = linalg.svd(data)
    print Sigma
    dataMat = mat(data)
    print euclidSim(dataMat[:, 0], dataMat[:, 4])
    print pearsSim(dataMat[:, 0], dataMat[:, 4])
    print cosSim(dataMat[:, 0], dataMat[:, 4])

    changeData(dataMat)
    print dataMat
    print recommend(dataMat, 2)

    dataMat2 = mat(loadExData2())
    U2, Sigma2, VT2 = linalg.svd(dataMat2)
    print Sigma2
    retCom = recommend(dataMat2, 1, estMethod=svdEst, simMeas=pearsSim)
    print retCom

    imgCompress(2)

if __name__ == '__main__':
    main()