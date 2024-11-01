from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pymongo import MongoClient
import re
import nltk

# Download stopwords (optional, if you need to use them later)
nltk.download('stopwords', quiet=True)

# Create or retrieve the existing SparkSession
spark = SparkSession.builder \
    .appName("SentimentAnalysisApp") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

# Load the data and handle any missing values
df = spark.read.csv("twitter_validation.csv", header=True, inferSchema=True)
df = df.na.fill({"Tweet-Comment": ""})  # Replace nulls with empty strings

# Improved clean_text_column function
def clean_text_column(df, column_name):
    return df \
        .withColumn(column_name, regexp_replace(col(column_name), r'https?://\S+|www\.\S+|\.com\S+|youtu\.be/\S+', '')) \
        .withColumn(column_name, regexp_replace(col(column_name), r'(@|#)\w+', '')) \
        .withColumn(column_name, regexp_replace(col(column_name), r'[^a-zA-Z\s]', '')) \
        .withColumn(column_name, regexp_replace(col(column_name), r'\s+', ' ')) \
        .withColumn(column_name, regexp_replace(col(column_name), r'^\s+|\s+$', ''))  # Trim whitespace

# Apply the cleaning function to the Tweet-Comment column
df = clean_text_column(df, "Tweet-Comment")

# Step 2: Tokenization
tokenizer = Tokenizer(inputCol="Tweet-Comment", outputCol="words")
wordsData = tokenizer.transform(df)

# Step 3: Compute the TF-IDF
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
featurizedData = hashingTF.transform(wordsData)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Remove existing 'features' column before applying VectorAssembler
rescaledData = rescaledData.drop("features")

# Assemble features using VectorAssembler
assembler = VectorAssembler(inputCols=['rawFeatures'], outputCol='features')
assembledData = assembler.transform(rescaledData)

# Convert Sentiment labels to numeric
assembledData = assembledData.withColumn("Sentiment_numeric",
                                         when(assembledData.Sentiment == "Positive", 1)
                                         .when(assembledData.Sentiment == "Negative", 0)
                                         .when(assembledData.Sentiment == "Neutral", 2)
                                         .when(assembledData.Sentiment == "Irrelevant", 3)
                                         .otherwise(None))

# Drop rows with null values in Sentiment_numeric
assembledData = assembledData.na.drop(subset=["Sentiment_numeric"])

# Split data into training and testing sets
trainData, testData = assembledData.randomSplit([0.8, 0.2], seed=1234)

# Train logistic regression model
lr = LogisticRegression(featuresCol='features', labelCol='Sentiment_numeric', maxIter=10)
lrModel = lr.fit(trainData)

# Make predictions on the test set
predictions = lrModel.transform(testData)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="Sentiment_numeric", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Test set accuracy = {accuracy:.2f}")

# Optional: Map predictions back to sentiment labels
predictions = predictions.withColumn("Predicted_Sentiment", 
                                     when(predictions.prediction == 0, "Negative")
                                     .when(predictions.prediction == 1, "Positive")
                                     .when(predictions.prediction == 2, "Neutral")
                                     .when(predictions.prediction == 3, "Irrelevant")
                                     .otherwise("Unknown"))

# Show sample predictions
predictions.select("Tweet-Comment", "Sentiment", "Predicted_Sentiment").show(10, truncate=False)

# Connect to MongoDB
mongo_client = MongoClient('mongodb://localhost:27017/')  # Update with your MongoDB connection string
db = mongo_client['twitter_sentiment']  # Database name
collection = db['predictions']  # Collection name

# Convert Spark DataFrame to Pandas DataFrame for easier insertion
predictions_pandas = predictions.select("Tweet-Comment", "Sentiment", "Predicted_Sentiment").toPandas()

# Insert predictions into MongoDB
records = predictions_pandas.to_dict(orient='records')
collection.insert_many(records)



# Helper function to clean text as a regular Python string (no Spark dependencies)
def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+|\.com\S+|youtu\.be/\S+', '', text)  # Remove URLs
    text = re.sub(r'(@|#)\w+', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()  # Trim leading and trailing spaces
    return text
class_index_mapping = {0: "Negative", 1: "Positive", 2: "Neutral", 3: "Irrelevant"}


# Define the classify_text function
def classify_text(text: str):
    # Clean the text using the helper function
    preprocessed_text = text

    # Create a DataFrame from the cleaned text
    data = [(preprocessed_text,)]
    data = spark.createDataFrame(data, ["Tweet-Comment"])

    # Apply transformations
    data = tokenizer.transform(data)
    data = hashingTF.transform(data)
    data = idfModel.transform(data)

    # Make the prediction
    prediction = lrModel.transform(data).select("prediction").collect()[0]["prediction"]

    # Map the prediction to a sentiment label
    predicted_sentiment = class_index_mapping[int(prediction)]

    # Print the details
    print("-> Tweet:", text)
    print("-> Preprocessed Tweet:", preprocessed_text)
    print("-> Predicted Sentiment:", predicted_sentiment)
    print("/" * 50)

    # Prepare and insert the document into MongoDB
    tweet_doc = {
        "tweet": text,
        "prediction": predicted_sentiment
    }
    collection.insert_one(tweet_doc)

    return predicted_sentiment

# Test the classify_text function
classify_text("I'm really happy with the new update!")
print("/" * 50)

# Stop Spark session
spark.stop()

