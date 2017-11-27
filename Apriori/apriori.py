#encoding:utf-8

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

# 从数据集中构建单元素的候选项集C1
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])   # C1为集合的集合
    C1.sort()
    return map(frozenset, C1)    # 对C1中的每个项构建一个不变集合

# 从候选项集Ck中筛选频繁项集Lk，并计算支持度
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can):   ssCnt[can] = 1
                else:   ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems     # 计算所有项集的支持度
        if support >= minSupport:
            retList.insert(0, key)          # 在列表的首部插入新的集合
        supportData[key] = support          # frozenset类型才能作为字典的key
    return retList, supportData

# 从频繁项集Lk(-1)中构建新的候选项集Ck
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
       for j in range(i + 1, lenLk):
           L1 = list(Lk[i])[:k-2]
           L2 = list(Lk[j])[:k-2]
           L1.sort()
           L2.sort()
           if L1 == L2:     # 前(k-2)个项相同时，将两个集合合并得到一个k元素的列表（集合）
               retList.append(Lk[i] | Lk[j])
    return retList

# 获取所有频繁项集的相应的支持度
def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)          # 候选项集C1
    D = map(set, dataSet)           # 数据映射为set类型
    L1, supportData = scanD(D, C1, minSupport)  # 频繁项集L1
    L = [L1]
    k = 2
    while len(L[k-2]) > 0:
        Ck = aprioriGen(L[k-2], k)             # 频繁项集构建候选项集
        Lk, supK = scanD(D, Ck, minSupport)    # 候选项集筛选频繁项集
        supportData.update(supK)          # 对存储支持度的字典进行更新
        L.append(Lk)
        k += 1
    return L, supportData

# 生成关联规则
def generateRules(L, supportData, minConf = 0.7):
    bigRuleList = []
    for i in range(1, len(L)):  # 只获取两个元素以上的集合
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]    # 规则后件元素个数从1到(k-1),从不符合最小置信度的子集排除超集
            if i > 1:
                ''' http://www.cnblogs.com/qwertWZ/p/4510857.html '''
                H1 = calcConf(freqSet, H1, supportData, bigRuleList, minConf)   # 必须先计算可信度（书本缺少这一行）！！！
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

'''
计算可信度
@freqSet 频繁项集 @H 所有可能的规则集合 @supportData 支持度
@bigRuleList 生成的规则集 @minConf 最小置信度
@return 筛选后的规则集
'''
def calcConf(freqSet, H, supportData, br1, minConf = 0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        print "No-->", freqSet - conseq, '-->', conseq, 'conf: ', conf
        if conf >= minConf:
            print freqSet - conseq, '-->', conseq, 'conf: ', conf
            br1.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

''' 频繁集元素个数k>=3时先进行切分(后件从1到k-1),得到后件后生成候选规则集合 '''
def rulesFromConseq(freqSet, H, supportData, br1, minConf = 0.7):
    m = len(H[0])
    if len(freqSet) > m + 1:
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)
        if len(Hmp1) > 1:
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)

# 收集美国国会议案中actionID的函数
def getActionID():
    from time import sleep
    from votesmart import votesmart
    votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
    actionIdList = []
    billTitleList = []
    fr = open('recent20bills.txt')
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum)
            for action in billDetail.actions:
                if action.level == 'House' and (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print 'bill: %d has actionId: %d' % (billNum, actionId)
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print "problem getting bill %d" % billNum
        sleep(1)
    return actionIdList, billTitleList

# 基于投票数据的事物列表填充函数
def getTransList(actionIdList, billTitleList):
    from time import sleep
    from votesmart import votesmart
    itemMeaning = ['Republican', 'Democratic']
    for billTitle in billTitleList:
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yes' % billTitle)
    transDict = {}
    voteCount = 2
    for actionId in actionIdList:   # 不同的法案条例对应不同的编码
        sleep(3)
        print 'Getting votes for actionId: %d' % actionId
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName):
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParities == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except:
            print "problem getting actionId: %d" % actionId
        voteCount += 2
    return transDict, itemMeaning



def main():
    # dataSet = loadDataSet()
    # L, supportData = apriori(dataSet, minSupport=0.5)
    # print L
    # bigRuleList = generateRules(L, supportData, minConf=0.7)
    # print bigRuleList

    # '''***********************************************'''
    # actionIdList, billTitleList = getActionID()
    # transDict, itemMeaning = getTransList(actionIdList, billTitleList)
    # voteDateSet = [transDict[key] for key in transDict.keys()]  # 构建包含所有事物项的列表
    # voteL, voteSupData = apriori(voteDateSet, minSupport=0.5)
    # voteRules = generateRules(voteL, voteSupData, minConf=0.7)

    '''************************************************'''
    mushDataSet = [line.split() for line in open('mushroom.dat').readlines()]
    mushL, mushSupData = apriori(mushDataSet, minSupport=0.3)
    for item in mushL[1]:   # 搜索包含特征2的频繁项集
        if item.intersection('2'):  print item
    for item in mushL[2]:
        if item.intersection('2'):  print item

if __name__ == '__main__':
    main()