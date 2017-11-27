from numpy import *
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    print type(fltLine)
    return dataMat


data = loadDataSet('ex00.txt')
data = mat(data)
print data[:, -1].T
print data[:, -1].T.tolist()
print data[:, -1].T.tolist()[0]
length = len(set(data[:, -1].T.tolist()[0]))
print length