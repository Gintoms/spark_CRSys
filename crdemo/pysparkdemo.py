from pyspark import SparkContext
import pyspark.mllib.recommendation as rd
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.evaluation import RegressionMetrics
import numpy

sc = SparkContext('local')

rawData = sc.textFile('u.data')
rawRatings = rawData.map(lambda line: line.split("\t")[:3])
ratings = rawRatings.map(lambda line: rd.Rating(int(line[0]), int(line[1]), float(line[2])))

model = ALS.train(ratings, 50, 10, 0.01)

predictedRating = model.predict(789,123)

topKRecs = model.recommendProducts(789,10)

for topKRec in topKRecs:
    print(topKRecs)

movies = sc.textFile('dataset/u.item')

titles_data= movies.map(lambda line: line.split("|")[:2]).collect()
titles = dict(titles_data)

moviesForUser = ratings.keyBy(lambda rating: rating.user).lookup(789)

moviesForUser = sorted(moviesForUser,key=lambda r: r.rating, reverse=True)[0:10]

for movieForUser in moviesForUser:
    print(movieForUser)

usersProducts = ratings.map(lambda r:(r.user, r.product))
predictions = model.predictAll(usersProducts).map(lambda r: ((r.user, r.product),r.rating))

ratingsAndPredictions = ratings.map(lambda r: ((r.user,r.product), r.rating)).join(predictions)


predictionsAndTrue = ratingsAndPredictions.map(lambda line: (line[1][0],line[1][3]))
regressionMetrics = RegressionMetrics(predictionsAndTrue)
print('MSE=%f' %regressionMetrics.meanSquaredError)
print('RMSE=%f' %regressionMetrics.rootMeanSquaredError)