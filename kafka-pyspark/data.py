from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Step 1: Create Spark Session
spark = SparkSession.builder \
    .appName("Simplified Logistic Regression") \
    .getOrCreate()

# Step 2: Load data
df = spark.read.csv("twitter_validation.csv", header=True, inferSchema=True)

# Step 3: Tokenize text data
tokenizer = Tokenizer(inputCol="Tweet-Comment", outputCol="words")
wordsData = tokenizer.transform(df)

# Step 4: Hashing Term Frequency
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
featurizedData = hashingTF.transform(wordsData)

# Step 5: IDF transformation
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Step 6: Split data into training and testing sets
trainData, testData = rescaledData.randomSplit([0.8, 0.2], seed=1234)

# Step 7: Train Logistic Regression model
lr = LogisticRegression(labelCol="Sentiment", featuresCol="features")
lrModel = lr.fit(trainData)

# Step 8: Make predictions on test data
predictions = lrModel.transform(testData)

# Step 9: Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="Sentiment", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Test Accuracy: {accuracy}")

# Stop Spark session
spark.stop()
