import re
import pandas as pd
import pymongo
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import findspark
findspark.init()
from nltk.corpus import stopwords
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import lower, regexp_replace
from pyspark.ml.feature import CountVectorizer, StringIndexer, Tokenizer, StopWordsRemover
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel, Pipeline, Transformer

def clean_text(df, inputCol="Text", outputCol="cleaned_text"):
    df = df.withColumn(outputCol, regexp_replace(df[inputCol], r'https?://\S+|www\.\S+|S+\.com\S+|youtu\.be/\S+', ''))
    df = df.withColumn(outputCol, regexp_replace(df[outputCol], r'(@|#)\w+', ''))
    df = df.withColumn(outputCol, lower(df[outputCol]))
    df = df.withColumn(outputCol, regexp_replace(df[outputCol], r'[^a-zA-Z\s]', ''))
    return df

label_mapping = {
    0: "Negative",
    1: "Positive",
    2: "Neutral",
    3: "Irrelevant"
}

label_mappings = {
    0: "Negative",
    1: "Positive",
    2: "Neutral",
    3: "Irrelevant"
}

def clean_new_text(df, inputCol="Text", outputCol="cleaned_text"):
    df = df.withColumn(outputCol, regexp_replace(df[inputCol], r'https?://\S+|www\.\S+|\.com\S+|youtu\.be/\S+', ''))
    df = df.withColumn(outputCol, regexp_replace(df[outputCol], r'(@|#)\w+', ''))
    df = df.withColumn(outputCol, lower(df[outputCol]))
    df = df.withColumn(outputCol, regexp_replace(df[outputCol], r'[^a-zA-Z\s]', ''))
    return df

def classify_text_lr(text: str):
    stop_words = stopwords.words('english')
    spark = SparkSession.builder \
        .appName("Text Classification with PySpark") \
        .getOrCreate()
    
    # Fetch data from MongoDB using pymongo
    mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')  
    db = mongo_client['twitter_sentiment']  
    collection = db['predictions']  
    
    # Convert MongoDB data to Pandas DataFrame
    data = list(collection.find())
    df = pd.DataFrame(data)
    df.columns = df.columns.str.strip()
    df['Sentiment_Numeric'] = df['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
    df['Predicted_Sentiment_Numeric'] = df['Predicted_Sentiment'].map({'Positive': 1, 'Negative': 0, 'Neutral': 2, 'Irrelevant': 3})
    
    # Handle missing values
    if df['Sentiment_Numeric'].isnull().any() or df['Predicted_Sentiment_Numeric'].isnull().any():
        df['Sentiment_Numeric'] = df['Sentiment_Numeric'].fillna(0)
        df['Predicted_Sentiment_Numeric'] = df['Predicted_Sentiment_Numeric'].fillna(0)

    # Drop the `_id` column and convert the DataFrame to a Spark DataFrame
    df = df.drop(columns=['_id'])
    spark_df = spark.createDataFrame(df)

    # StringIndexer for the Sentiment column
    label_indexer = StringIndexer(inputCol="Sentiment", outputCol="Label2")
    label_indexer_model = label_indexer.fit(spark_df)
    spark_df = label_indexer_model.transform(spark_df)

    # Extract label mappings
    label_mappings = {i: label for i, label in enumerate(label_indexer_model.labels)}

    # Drop rows with empty 'Tweet-Comment' column
    spark_df = spark_df.dropna(subset=['Tweet-Comment'])

    # Clean and preprocess the data
    cleaned_data = clean_text(spark_df, inputCol="Tweet-Comment", outputCol="Text")

    # Tokenizer, StopWordsRemover, CountVectorizer
    tokenizer = Tokenizer(inputCol="Text", outputCol="tokens")
    stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", stopWords=stop_words)
    count_vectorizer = CountVectorizer(inputCol="filtered_tokens", outputCol="features", vocabSize=10000, minDF=5)

    # Logistic Regression
    lr = LogisticRegression(maxIter=10, labelCol="Label2", featuresCol="features")

    # Pipeline
    pipeline = Pipeline(stages=[tokenizer, stopwords_remover, count_vectorizer, lr])

    # Fit the pipeline
    model = pipeline.fit(cleaned_data)

    # Accuracy Evaluation
    evaluator = MulticlassClassificationEvaluator(labelCol="Label2", predictionCol="prediction", metricName="accuracy")
    processed_data = model.transform(cleaned_data)
    accuracy = evaluator.evaluate(processed_data)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Classify new text
    new_text_df = spark.createDataFrame([Row(**{'Tweet-Comment': text})])
    cleaned_new_text = clean_text(new_text_df, inputCol="Tweet-Comment", outputCol="Text")
    predictions = model.transform(cleaned_new_text)
    result = predictions.select("Tweet-Comment", "prediction").collect()
    predicted_label = label_mappings[int(result[0]['prediction'])]
    
    return predicted_label


def classify_text_nb(text: str):
    stop_words = stopwords.words('english')
    spark = SparkSession.builder \
        .appName("Text Classification with PySpark") \
        .getOrCreate()
    mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')  
    db = mongo_client['twitter_sentiment']  
    collection = db['predictions']  
    data = list(collection.find())
    df = pd.DataFrame(data)
    df.columns = df.columns.str.strip()
    df['Sentiment_Numeric'] = df['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
    df['Predicted_Sentiment_Numeric'] = df['Predicted_Sentiment'].map({'Positive': 1, 'Negative': 0, 'Neutral': 2, 'Irrelevant': 3})
    if df['Sentiment_Numeric'].isnull().any() or df['Predicted_Sentiment_Numeric'].isnull().any():
        df['Sentiment_Numeric'] = df['Sentiment_Numeric'].fillna(0)
        df['Predicted_Sentiment_Numeric'] = df['Predicted_Sentiment_Numeric'].fillna(0)
    df = df.drop(columns=['_id'])
    spark_df = spark.createDataFrame(df)
    label_indexer = StringIndexer(inputCol="Sentiment", outputCol="Label2")
    label_indexer_model = label_indexer.fit(spark_df)
    spark_df = label_indexer_model.transform(spark_df)
    label_mappings = {i: label for i, label in enumerate(label_indexer_model.labels)}
    spark_df = spark_df.dropna(subset=['Tweet-Comment'])
    tokenizer = Tokenizer(inputCol="Tweet-Comment", outputCol="tokens")
    stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", stopWords=stop_words)
    count_vectorizer = CountVectorizer(inputCol="filtered_tokens", outputCol="features", vocabSize=10000, minDF=5)
    nb = NaiveBayes(labelCol="Label2", featuresCol="features")
    pipeline = Pipeline(stages=[tokenizer, stopwords_remover, count_vectorizer, nb])
    model = pipeline.fit(spark_df)
    evaluator = MulticlassClassificationEvaluator(labelCol="Label2", predictionCol="prediction", metricName="accuracy")
    processed_data = model.transform(spark_df)
    accuracy = evaluator.evaluate(processed_data)
    new_text_df = spark.createDataFrame([Row(**{'Tweet-Comment': text})])
    predictions = model.transform(new_text_df)
    result = predictions.select("Tweet-Comment", "prediction").collect()
    predicted_label = label_mappings[int(result[0]['prediction'])]
    return predicted_label


def classify_text_rf(text: str):
    stop_words = stopwords.words('english')
    spark = SparkSession.builder \
        .appName("Text Classification with PySpark") \
        .getOrCreate()
    
    # Fetch data from MongoDB using pymongo
    mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')  
    db = mongo_client['twitter_sentiment']  
    collection = db['predictions']  
    
    # Convert MongoDB data to Pandas DataFrame
    data = list(collection.find())
    df = pd.DataFrame(data)
    df.columns = df.columns.str.strip()
    df['Sentiment_Numeric'] = df['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
    df['Predicted_Sentiment_Numeric'] = df['Predicted_Sentiment'].map({'Positive': 1, 'Negative': 0, 'Neutral': 2, 'Irrelevant': 3})
    
    # Handle missing values
    if df['Sentiment_Numeric'].isnull().any() or df['Predicted_Sentiment_Numeric'].isnull().any():
        df['Sentiment_Numeric'] = df['Sentiment_Numeric'].fillna(0)
        df['Predicted_Sentiment_Numeric'] = df['Predicted_Sentiment_Numeric'].fillna(0)

    # Drop the `_id` column and convert the DataFrame to a Spark DataFrame
    df = df.drop(columns=['_id'])
    spark_df = spark.createDataFrame(df)

    # StringIndexer for the Sentiment column
    label_indexer = StringIndexer(inputCol="Sentiment", outputCol="Label2")
    label_indexer_model = label_indexer.fit(spark_df)
    spark_df = label_indexer_model.transform(spark_df)

    # Extract label mappings
    label_mappings = {i: label for i, label in enumerate(label_indexer_model.labels)}

    # Drop rows with empty 'Tweet-Comment' column
    spark_df = spark_df.dropna(subset=['Tweet-Comment'])

    # Clean and preprocess the data
    cleaned_data = clean_text(spark_df, inputCol="Tweet-Comment", outputCol="Text")

    # Tokenizer, StopWordsRemover, CountVectorizer
    tokenizer = Tokenizer(inputCol="Text", outputCol="tokens")
    stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", stopWords=stop_words)
    count_vectorizer = CountVectorizer(inputCol="filtered_tokens", outputCol="features", vocabSize=10000, minDF=5)

    # Random Forest Classifier
    rf = RandomForestClassifier(labelCol="Label2", featuresCol="features", numTrees=20)

    # Pipeline
    pipeline = Pipeline(stages=[tokenizer, stopwords_remover, count_vectorizer, rf])

    # Fit the pipeline
    model = pipeline.fit(cleaned_data)

    # Accuracy Evaluation
    evaluator = MulticlassClassificationEvaluator(labelCol="Label2", predictionCol="prediction", metricName="accuracy")
    processed_data = model.transform(cleaned_data)
    accuracy = evaluator.evaluate(processed_data)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Classify new text
    new_text_df = spark.createDataFrame([Row(**{'Tweet-Comment': text})])
    cleaned_new_text = clean_text(new_text_df, inputCol="Tweet-Comment", outputCol="Text")
    predictions = model.transform(cleaned_new_text)
    result = predictions.select("Tweet-Comment", "prediction").collect()
    predicted_label = label_mappings[int(result[0]['prediction'])]
    
    return predicted_label



def classify_text_dt(text: str):
    stop_words = stopwords.words('english')
    spark = SparkSession.builder \
        .appName("Text Classification with PySpark") \
        .getOrCreate()
    
    # Fetch data from MongoDB using pymongo
    mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')  
    db = mongo_client['twitter_sentiment']  
    collection = db['predictions']  
    
    # Convert MongoDB data to Pandas DataFrame
    data = list(collection.find())
    df = pd.DataFrame(data)
    df.columns = df.columns.str.strip()
    df['Sentiment_Numeric'] = df['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
    df['Predicted_Sentiment_Numeric'] = df['Predicted_Sentiment'].map({'Positive': 1, 'Negative': 0, 'Neutral': 2, 'Irrelevant': 3})
    
    # Handle missing values
    if df['Sentiment_Numeric'].isnull().any() or df['Predicted_Sentiment_Numeric'].isnull().any():
        df['Sentiment_Numeric'] = df['Sentiment_Numeric'].fillna(0)
        df['Predicted_Sentiment_Numeric'] = df['Predicted_Sentiment_Numeric'].fillna(0)

    # Drop the `_id` column and convert the DataFrame to a Spark DataFrame
    df = df.drop(columns=['_id'])
    spark_df = spark.createDataFrame(df)

    # StringIndexer for the Sentiment column
    label_indexer = StringIndexer(inputCol="Sentiment", outputCol="Label2")
    label_indexer_model = label_indexer.fit(spark_df)
    spark_df = label_indexer_model.transform(spark_df)

    # Extract label mappings
    label_mappings = {i: label for i, label in enumerate(label_indexer_model.labels)}

    # Drop rows with empty 'Tweet-Comment' column
    spark_df = spark_df.dropna(subset=['Tweet-Comment'])

    # Clean and preprocess the data
    cleaned_data = clean_text(spark_df, inputCol="Tweet-Comment", outputCol="Text")

    # Tokenizer, StopWordsRemover, CountVectorizer
    tokenizer = Tokenizer(inputCol="Text", outputCol="tokens")
    stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", stopWords=stop_words)
    count_vectorizer = CountVectorizer(inputCol="filtered_tokens", outputCol="features", vocabSize=10000, minDF=5)

    # Decision Tree Classifier
    dt = DecisionTreeClassifier(labelCol="Label2", featuresCol="features")

    # Pipeline
    pipeline = Pipeline(stages=[tokenizer, stopwords_remover, count_vectorizer, dt])

    # Fit the pipeline
    model = pipeline.fit(cleaned_data)

    # Accuracy Evaluation
    evaluator = MulticlassClassificationEvaluator(labelCol="Label2", predictionCol="prediction", metricName="accuracy")
    processed_data = model.transform(cleaned_data)
    accuracy = evaluator.evaluate(processed_data)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Classify new text
    new_text_df = spark.createDataFrame([Row(**{'Tweet-Comment': text})])
    cleaned_new_text = clean_text(new_text_df, inputCol="Tweet-Comment", outputCol="Text")
    predictions = model.transform(cleaned_new_text)
    result = predictions.select("Tweet-Comment", "prediction").collect()
    predicted_label = label_mappings[int(result[0]['prediction'])]
    
    return predicted_label
