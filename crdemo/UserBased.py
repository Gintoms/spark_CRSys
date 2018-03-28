import math

class UserBased:
    def __init__(self, userData, k):
        self.userData = userData
        self.k = k

    # 计算两个用户之间的相似度
    def simPerson(self, user1, user2):
        data = self.userData
        sim = {}
        # 若用户1、2同时对一个物品进行了评分，则记录下来
        for item in data[user1]:
            if item in data[user2]:
                sim[item] = 1
        n = len(sim)
        if not n:
            return -1
        # 求和
        sum1 = sum([data[user1][item] for item in sim])
        sum2 = sum([data[user2][item] for item in sim])
        # 求和的平方
        sum1Sq = sum([math.pow(data[user1][item], 2) for item in sim])
        sum2Sq = sum([math.pow(data[user2][item], 2) for item in sim])
        # 求乘积之和
        sumMulti = sum([data[user1][item] * data[user2][item] for item in sim])
        num1 = sumMulti - sum1 * sum2 / n
        num2 = math.sqrt((sum1Sq - math.pow(sum1, 2) / n) * (sum2Sq - math.pow(sum2, 2) / n))
        if not num2:
            return -1
        return 0.5 + 0.5 * (num1 / num2)

    def kNeibors(self, theUserID, k):
        data = self.userData
        # 计算其他用户和改用户的相似度
        # 其他用户的ID 相似度
        similarities = [(otherID, self.simPerson(theUserID, otherID)) for otherID in data if otherID != theUserID]
        # 根据相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        # 返回相似度排名前k的用户集合
        return similarities[0: k]

    # 计算预测评分
    def estimatePref(self, theUserID, theItemID, simUsers=None):
        data = self.userData
        try:
            truePref = data[theUserID][theItemID]
        except KeyError:
            truePref = 0
        if truePref:
            return truePref
        total = 0.0
        simSum = 0.0
        simUsers = simUsers or self.kNeibors(theUserID, self.k)
        for otherID, sim in simUsers:
            if sim <= 0: continue
            try:
                # 获取其他用户对该商品的评分
                otherTruePref = data[otherID][theItemID]
            except KeyError:
                continue
            total += otherTruePref * sim
            simSum += sim
        if not simSum:
            return -1
        # 返回相似用户中对某物品的加权评分
        return total / simSum

    def recommend(self, theUserID, howMany):
        data = self.userData
        kNeighbors = self.kNeibors(theUserID, self.k)
        ranks = []
        for otherID, in kNeighbors:
            tempRanks = [(itemID, self.estimatePref(theUserID, itemID, kNeighbors)) for itemID in data[otherID] if
                         itemID not in data[theUserID]]
            ranks.extend(tempRanks)
        ranks.sort(lambda x: x[1])
        return ranks[: -(howMany + 1): -1]
