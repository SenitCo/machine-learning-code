#encoding:utf-8

# FP树的类定义
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur       # 元素项的个数
        self.nodeLink = None        # 链接相似的元素项
        self.parent = parentNode    # 指向父结点
        self.children = {}          # 存放子结点

    def inc(self, numOccur):    # 对count变量增加给定值
        self.count += numOccur

    def disp(self, ind=1):      # 将树以文本形式显示
        print '  ' * ind, self.name, ' ', self.count
        for child in self.children.values():
            child.disp(ind + 1)


# FP树构建函数
def createTree(dataSet, minSup = 1):
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]   # 对元素项计数
    for k in headerTable.keys():
        if headerTable[k] < minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    print headerTable
    for k in headerTable:   # 取key值
        headerTable[k] = [headerTable[k], None]     # 将字典的key值作为索引链接元素项<头指针表>
    print headerTable
    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        # print tranSet, count
        localD = {}
        for item in tranSet:
            if item in freqItemSet:     # 筛选在频繁项集出现的元素项
                localD[item] = headerTable[item][0]     # 元素项出现的次数(==get(item, 0))
                # print item, '-->', localD[item]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]  # 按大小排序
            # print orderedItems
            updateTree(orderedItems, retTree, headerTable, count)   # 使用排序后的频繁项集对树进行填充
    return retTree, headerTable

'''
纵向构建FP树
@items 排序后的频繁项集 @inTree 父结点
@ headerTable 头指针表 @count 计数值
'''
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)    # 如果结点已存在则累加计数
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)   # 不存在则创建新的结点
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]    # 如果相应头指针表的指针为空则链接该结点
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])   # 否则结点链接到链表尾
    # print items
    if len(items) > 1:  # 向FP树中依次添加或更新频繁项集（均为单元素）的元素项
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

'''
横向创建相似项的链表
@nodeToTest 头指针表中指向元素项的指针
@targetNode 待添加到链表尾的结点
'''
def updateHeader(nodeToTest, targetNode):
    while nodeToTest.nodeLink != None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def loadSimpData():
    simpData = [['r', 'z', 'h', 'j', 'p'],
                ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                ['z'],
                ['r', 'x', 'n', 'o', 's'],
                ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpData

# 将列表转换为字典，且确定相应键值的频率
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

'''
迭代上溯FP树，确定路径
@leafNode 给定子结点  @prefixPath 存储子结点的向上路径
'''
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

'''
搜索给定元素项的向上路径，抽取条件模式基
@basePat 给定元素项 @treeNode 元素项对应的结点
@return condPats 条件模式基和元素项结点的计数值
'''
def findPrefixPath(basePat, treeNode):
    conPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            conPats[frozenset(prefixPath[1:])] = treeNode.count  # 不包含给点结点
        treeNode = treeNode.nodeLink
    return conPats

'''
创建条件FP树，递归查找频繁项集
@inTree FP树 @ headerTable 头指针表 @ 最小支持度
@preFix 频繁项 @freqIemList 存储所有频繁项的频繁项集
'''
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]  # 对头指针表排序
    for basePat in bigL:    # 遍历头指针表中的元素项
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        condTree, condHT = createTree(condPattBases, minSup)    # 利用条件模式基构建条件FP树
        if condHT != None:
            print "conditional tree for: ", newFreqSet
            condTree.disp()
            mineTree(condTree, condHT, minSup, newFreqSet, freqItemList)    # 递归挖掘条件FP树

# 访问Twitter Python库的代码
import twitter
from time import sleep
import re

def textParse(bigString):
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)
    listOfTokens = re.split(r'\W*', urlsRemoved)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def getLotsOfTweets(searchStr):
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    ACCESS_TOKEN_KEY = ''
    ACCESS_TOKEN_SECRET = ''
    api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY,
                      access_token_secret=ACCESS_TOKEN_SECRET)
    #you can get 1500 results 15 pages * 100 per page
    resultsPages = []
    for i in range(1,15):
        print "fetching page %d" % i
        searchResults = api.GetSearch(searchStr, per_page=100, page=i)
        resultsPages.append(searchResults)
        sleep(6)
    return resultsPages

def mineTweets(tweetArr, minSup=5):
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))
    initSet = createInitSet(parsedList)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    return myFreqList

def main():
    simpData = loadSimpData()
    initSet = createInitSet(simpData)
    retTree, headerTable = createTree(initSet, 3)
    retTree.disp()
    freqItemList = []
    mineTree(retTree, headerTable, 3, set([]), freqItemList)
    print freqItemList

    # 从kosarak.dat文件中导入百万数据记录
    parseData = [line.split() for line in open('kosarak.dat').readlines()]
    initDataSet = createInitSet(parseData)
    fpTree, headerTab = createTree(initDataSet, minSup=100000)
    freqList = []
    mineTree(fpTree, headerTab, 100000, set([]), freqList)
    print freqList

if __name__ == "__main__":
    main()