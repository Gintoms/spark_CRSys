from pyspark import SparkContext
from pyspark.mllib.recommendation import Rating, ALS

sc = SparkContext('local')

def loadFile():
    rawData = sc.textFile('dataset/u.data')
    rawRatings = rawData.map(lambda x: x.split('\t'))
    return rawRatings

if __name__ == '__main__':

    loadFile()