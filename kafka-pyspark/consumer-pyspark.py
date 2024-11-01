from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder \
    .appName("Simplified Logistic Regression") \
    .getOrCreate()

df = spark.read.csv("twitter_validation.csv", header=True, inferSchema=True)

tokenizer = Tokenizer(inputCol="text_column", outputCol="words")
wordsData = tokenizer.transform(df)

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
featurizedData = hashingTF.transform(wordsData)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

trainData, testData = rescaledData.randomSplit([0.8, 0.2], seed=1234)

# Step 5: Train a Logistic Regression model
lr = LogisticRegression(labelCol="label_column", featuresCol="features")
lrModel = lr.fit(trainData)

# Step 6: Make predictions on the test data
predictions = lrModel.transform(testData)

# Step 7: Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="label_column", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Test Accuracy: {accuracy}")

# Stop Spark session
spark.stop()
