import time
import math
import random
from threading import Thread
from threading import Lock
import logging

from crdemo.UserBased import UserBased
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='cr.log', level=logging.DEBUG, format=LOG_FORMAT)

class Evaluator:
    def __init__(self):
        self.diSum = 0.0
        self.count = 0
        self.lock = Lock()

    # testPercentage 表示测试数据和训练数据的比例
    def evaluate(self, data, testPercentage):
        self.data = data
        self.testPercentage = testPercentage
        startTime = time.clock()
        testPercentage = testPercentage or self.testPercentage
        # 根据testPercentage分割数据集，分为训练集和检验集
        trainData, testData = self.splitData(self.data, self.testPercentage)
        # 训练
        self.recommender = UserBased(trainData, 10)
        # 将验证集分为3部分
        # part1Data, part2Data, part3Data = self.splitTestDataTo3Parts(testData)
        # 开启三个线程计算RMSE值
        t1 = Thread(target=self.doEvaluate, args=(trainData, testData))
        t1.start()
        t1.join()
        result = math.sqrt(self.diSum / self.count)
        print('计算RMSE结束, RMSE值为: %s; 用时: %s 秒' % (result, time.clock() - startTime))
        logging.debug('计算RMSE结束, RMSE值为: %s; 用时: %s 秒' % (result, time.clock() - startTime))
        return result

    def splitData(self, data=None, testPercentage=None):
        data = data or self.data
        testPerc = testPercentage or self.testPercentage
        # 分别是训练集和检验集
        trainData = {}
        testData = {}
        for user in data:
            for item, score in data[user].items():
                if random.random() < testPerc:
                    testData.setdefault(user, {})
                    testData[user][item] = score
                else:
                    trainData.setdefault(user, {})
                    trainData[user][item] = score
        return trainData, testData

    def splitTestDataTo3Parts(self, testData):
        part1Data = {}
        part2Data = {}
        part3Data = {}
        for user in testData:
            x = random.random()
            if x < 0.3:
                part1Data[user] = testData[user]
            elif x < 0.6:
                part2Data[user] = testData[user]
            else:
                part3Data[user] = testData[user]
        return part1Data, part2Data, part3Data

    def doEvaluate(self, trainData, partTestData):
        partDiSum = 0.0
        partCount = 0
        recommender = self.recommender
        k = recommender.k
        for user in partTestData:
            # 与当前用户最相近的k个用户
            simUsers = recommender.kNeibors(user, k)
            print('userID=%i' %user)
            logging.debug('userID=%i' %user)
            for item, score in partTestData[user].items():
                predictPref = recommender.estimatePref(user, item, simUsers)
                if predictPref < 0: continue
                for simUser in simUsers:
                    print('相似用户有%i,相似度为%f' %(simUser[0], simUser[1]))
                    logging.debug('相似用户有%i,相似度为%f' %(simUser[0], simUser[1]))
                print('用户%i对%i物品的预测' %(user, item))
                logging.debug('用户%i对%i物品的预测' %(user, item))
                print('预测值为: %i' %predictPref)
                logging.debug('预测值为: %i' %predictPref)
                print('真实值为: %i' %score)
                logging.debug('真实值为: %i' %score)
                partDiSum += math.pow(predictPref - score, 2)
                partCount += 1
        self.lock.acquire()
        self.diSum += partDiSum
        self.count += partCount
        self.lock.release()


# 加载数据
def loadDate(filename):
    startTime = time.clock()
    totalData = {}
    count = 0
    for line in open(filename):
        rowData = line.split('\t')
        userID = rowData[0]
        itemID = rowData[1]
        score = rowData[2]
        # userID, itemID, score = line.split('\t')
        user, item, score = int(userID), int(itemID), int(score)
        # 将totalData的格式设置成key->map
        totalData.setdefault(user, {})
        # 设置totalData中用户对物品的评分数据
        totalData[user][item] = score
        count += 1
    print('数据加载成功! 用时: %s秒 总记录: %s 行,用户数: %s' % (time.clock() - startTime, count, len(totalData)))
    logging.debug('数据加载成功! 用时: %s秒 总记录: %s 行,用户数: %s' % (time.clock() - startTime, count, len(totalData)))
    return totalData


if __name__ == '__main__':
    # 加载文件
    data = loadDate('./dataset/u.data')
    Evaluator().evaluate(data, 0.3)
