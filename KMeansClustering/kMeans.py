#encoding:utf-8

from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine= line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat

# 计算欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

# 随机选取k个聚类中心点
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)     # k维数组
    return centroids

# KMeans聚类
def kMeans(dataSet, k, distMeas = distEclud, createCent = randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))   # 第一列记录簇索引值，第二列存储到质心的距离(平方)
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:    # 确定最近的质心
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:    # 质心发生改变则继续迭代
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        #print centroids
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

# 二分K-均值聚类算法
def biKMeans(dataSet, k, distMeas = distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]      # 转换成列表形式
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):      # nonzero用于过滤数组元素
            ptsIncurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsIncurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print "sseSplit: %d and sseNotSplit: %d " % (sseSplit, sseNotSplit)
            if(sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCent = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print 'The bestCentToSplit is: ', bestCentToSplit
        print 'The length if bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCent[0, :].tolist()[0]
        ''' 要转换成列表形式<tolist.()[0]>!!!,否则centList为列表，元素为矩阵，以mat()返回会报错 '''
        centList.append(bestNewCent[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssment

''' Test: 对地图上的点进行聚类 '''
# Yahoo! PlaceFinder
def geoGrab(stAddress, city):
    import urllib
    import json 
    apiStem = 'http://where.yahooapis.com/geocode?'
    params = {}
    params['flags'] = 'J'
    params['appid'] = 'ppp68N8t'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params
    print yahooApi
    c = urllib.urlopen(yahooApi)
    return json.loads(c.read())

# 获取经纬度
def massPlaceFind(fileName):
    from time import sleep
    fw = open('places1.txt', 'w')
    for line in open(fileName).readlines():
        lineArr = line.strip().split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print "%s\t%f\t%f" % (lineArr[0], lat, lng)
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else:
            print "Error fetching"
        sleep(1)
    fw.close()

# 球面距离计算
''' C = sin(LatA*Pi/180)*sin(LatB*Pi/180) + cos(LatA*Pi/180)*cos(LatB*Pi/180)*cos((MLonA-MLonB)*Pi/180)
Distance = R*Arccos(C)*Pi/180 <经纬度:A(LonA, latA), B(LonB, LatB)> '''
def distSLC(vecA, vecB):
    Rad = 6371.0      # 地球平均半径6371km，赤道半径6379km，极半径6357km
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * cos((vecB[0, 0] -vecA[0, 0]) * pi / 180)
    return arccos(a + b) * Rad

# 地理坐标聚类及簇绘制函数
def clusterGeoCoord(numClust = 5):
    import matplotlib
    import matplotlib.pyplot as plt
    dataList = []
    fr = open('places.txt')
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataList.append([float(lineArr[4]), float(lineArr[3])])
    dataMat = mat(dataList)
    centroids, clusterAss = biKMeans(dataMat, numClust, distMeas=distSLC)
    centroids = mat(centroids)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMakers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = dataMat[nonzero(clusterAss[:, 0].A == i)[0], :]
        markerStyle = scatterMakers[i % len(scatterMakers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0],
                    marker=markerStyle, s=90)
    ax1.scatter(centroids[:, 0].flatten().A[0], centroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()

def main():
    # dataMat = mat(loadDataSet('testSet.txt'))
    # centroidsMat, clustAssMat = kMeans(dataMat, 4)
    # print centroidsMat
    # dataMat2 = mat(loadDataSet('testSet2.txt'))
    # centList, clustAss = biKMeans(dataMat2, 3)
    # print centList
    # massPlaceFind('portlandClubs.txt')    ?????
    clusterGeoCoord(5)

if __name__ == '__main__':
    main()