import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import findspark
findspark.init()
from nltk.corpus import stopwords
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, pandas_udf, col, lower, regexp_replace
from pyspark.ml.feature import CountVectorizer, StringIndexer, Tokenizer, StopWordsRemover
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Define English stopwords
stop_words = stopwords.words('english')

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Text Classification with Logistic Regression") \
    .getOrCreate()

# Load training and validation datasets
data = spark.read.csv('twitter_training.csv', header=False, inferSchema=True)
validation = spark.read.csv('twitter_validation.csv', header=False, inferSchema=True)

# Define column names
columns = ['id', 'Company', 'Label', 'Text']

# Rename columns
for i, col_name in enumerate(columns):
    data = data.withColumnRenamed(f'_c{i}', col_name)
    validation = validation.withColumnRenamed(f'_c{i}', col_name)

# Define the StringIndexer for the label column
label_indexer = StringIndexer(inputCol="Label", outputCol="Label2")

# Drop rows with empty 'Text' column
data = data.dropna(subset=['Text'])
validation = validation.dropna(subset=['Text'])

# Fit StringIndexer on data
label_indexer_model = label_indexer.fit(data)
data = label_indexer_model.transform(data)
validation = label_indexer_model.transform(validation)

# Extract label mapping
label_mapping = label_indexer_model.labels

# Print label mapping
print("Label Mapping:")
for index, label in enumerate(label_mapping):
    print(f"Index {index} --> Label '{label}'")

def clean_text(df, inputCol="Text", outputCol="cleaned_text"):
    df = df.withColumn(outputCol, regexp_replace(df[inputCol], r'https?://\S+|www\.\S+|S+\.com\S+|youtu\.be/\S+', ''))
    df = df.withColumn(outputCol, regexp_replace(df[outputCol], r'(@|#)\w+', ''))
    df = df.withColumn(outputCol, lower(df[outputCol]))
    df = df.withColumn(outputCol, regexp_replace(df[outputCol], r'[^a-zA-Z\s]', ''))
    return df

# Clean text data
cleaned_data = clean_text(data, inputCol="Text", outputCol="Text")
cleaned_validation = clean_text(validation, inputCol="Text", outputCol="Text")

# Define tokenizer
tokenizer = Tokenizer(inputCol="Text", outputCol="tokens")

# Define stopwords remover
stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", stopWords=stop_words)

# Define CountVectorizer
count_vectorizer = CountVectorizer(inputCol="filtered_tokens", outputCol="features", vocabSize=10000, minDF=5)

# Define Logistic Regression
lr = LogisticRegression(labelCol="Label2", featuresCol="features")

# Create the pipeline
pipeline = Pipeline(stages=[tokenizer, stopwords_remover, count_vectorizer, lr])

# Apply the pipeline to the data
model = pipeline.fit(cleaned_data)
processed_data = model.transform(cleaned_data)

# Show the processed data with predictions
processed_data.select("Text", "Label2", "prediction").show()
evaluator = MulticlassClassificationEvaluator(labelCol="Label2", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(processed_data)

# Print the total number of rows and accuracy
total_rows = processed_data.count()
print(f"Total number of rows: {total_rows}")
print(f"Accuracy of the model: {accuracy:.2f}")

def clean_new_text(df, inputCol="Text", outputCol="cleaned_text"):
    df = df.withColumn(outputCol, regexp_replace(df[inputCol], r'https?://\S+|www\.\S+|\.com\S+|youtu\.be/\S+', ''))
    df = df.withColumn(outputCol, regexp_replace(df[outputCol], r'(@|#)\w+', ''))
    df = df.withColumn(outputCol, lower(df[outputCol]))
    df = df.withColumn(outputCol, regexp_replace(df[outputCol], r'[^a-zA-Z\s]', ''))
    return df

def classify_text(text):
    new_text_df = spark.createDataFrame([('company1', text)], ["Company", "Text"])
    cleaned_new_text = clean_new_text(new_text_df, inputCol="Text", outputCol="Text")
    predictions = model.transform(cleaned_new_text)
    result = predictions.select("Text", "prediction").collect()
    return result

# Call the classify_text function with sample inputs
sample_texts = [
    "This is a great product!",
    "I didn't like the service.",
    "Neutral comment about the product.",
    "Not relevant comment that doesn't matter."
]

for text in sample_texts:
    prediction = classify_text(text)
    print(f"Text: '{text}' | Prediction: {prediction[0]['prediction']}")
