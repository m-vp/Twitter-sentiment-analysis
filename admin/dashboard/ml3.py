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
    """Clean the new text data."""
    df = df.withColumn(outputCol, regexp_replace(df[inputCol], r'https?://\S+|www\.\S+|\.com\S+|youtu\.be/\S+', ''))
    df = df.withColumn(outputCol, regexp_replace(df[outputCol], r'(@|#)\w+', ''))
    df = df.withColumn(outputCol, lower(df[outputCol]))  # Convert text to lowercase
    df = df.withColumn(outputCol, regexp_replace(df[outputCol], r'[^a-zA-Z\s]', ''))  # Remove non-alpha characters
    return df

def classify_text_lr(text: str):
    stop_words = stopwords.words('english')

    # Create a SparkSession
    spark = SparkSession.builder \
        .appName("Text Classification with PySpark") \
        .getOrCreate()

    # Load training and validation datasets
    data = spark.read.csv('twitter_training.csv', header=False, inferSchema=True)
    validation = spark.read.csv('twitter_validation.csv', header=False, inferSchema=True)

    # Define column names
    columns = ['id', 'Company', 'Label', 'Text']

    for i, col_name in enumerate(columns):
        data = data.withColumnRenamed(f'_c{i}', col_name)
        validation = validation.withColumnRenamed(f'_c{i}', col_name)

    label_indexer = StringIndexer(inputCol="Label", outputCol="Label2")

    data = data.dropna(subset=['Text'])
    validation = validation.dropna(subset=['Text'])


    label_indexer = StringIndexer(inputCol="Label", outputCol="Label2")


    label_indexer_model = label_indexer.fit(data)
    data = label_indexer_model.transform(data)
    validation = label_indexer_model.transform(validation)

    # Extract label mapping
    label_mapping = label_indexer_model.labels

    # Print label mapping
    print("Label Mapping:")
    for index, label in enumerate(label_mapping):
        print(f"Index {index} --> Label '{label}'")
        
        
    cleaned_data = clean_text(data, inputCol="Text", outputCol="Text")
    cleaned_validation = clean_text(validation, inputCol="Text", outputCol="Text")

    # Define tokenizer
    tokenizer = Tokenizer(inputCol="Text", outputCol="tokens")

    # Define stopwords remover
    stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", stopWords=stop_words)

    # Define CountVectorizer
    count_vectorizer = CountVectorizer(inputCol="filtered_tokens", outputCol="features", vocabSize=10000, minDF=5)

    # Define Logistic Regression
    lr = LogisticRegression(maxIter=10, labelCol="Label2", featuresCol="features")

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

    
    """Classify the sentiment of a given text input."""
    new_text_df = spark.createDataFrame([('company1', text)], ["Company", "Text"])  

    cleaned_new_text = clean_new_text(new_text_df, inputCol="Text", outputCol="Text")

    predictions = model.transform(cleaned_new_text)

    result = predictions.select("Text", "prediction").collect()

    sentiment_label = label_mappings.get(result[0]['prediction'], "Unknown")
    
    if result:  
        prediction_index = result[0]['prediction']  
        sentiment_label = label_mappings.get(prediction_index, "Unknown")
        return sentiment_label
    else:
        return "No prediction available"



def classify_text_nb(text: str):
    stop_words = stopwords.words('english')

    # Create a SparkSession
    spark = SparkSession.builder \
        .appName("Text Classification with PySpark") \
        .getOrCreate()

    # Fetch data from MongoDB using pymongo
    mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')  # MongoDB connection
    db = mongo_client['twitter_sentiment']  # Database name
    collection = db['predictions']  # Collection name

    # Convert MongoDB data to Pandas DataFrame
    data = list(collection.find())
    df = pd.DataFrame(data)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Map sentiments to numeric values
    df['Sentiment_Numeric'] = df['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
    df['Predicted_Sentiment_Numeric'] = df['Predicted_Sentiment'].map({'Positive': 1, 'Negative': 0, 'Neutral': 2, 'Irrelevant': 3})

    # Handle missing values
    if df['Sentiment_Numeric'].isnull().any() or df['Predicted_Sentiment_Numeric'].isnull().any():
        print("NaN values found in Sentiment_Numeric or Predicted_Sentiment_Numeric")
        df['Sentiment_Numeric'] = df['Sentiment_Numeric'].fillna(0)
        df['Predicted_Sentiment_Numeric'] = df['Predicted_Sentiment_Numeric'].fillna(0)

    # Drop the `_id` column and convert the DataFrame to a Spark DataFrame
    df = df.drop(columns=['_id'])
    spark_df = spark.createDataFrame(df)

    # StringIndexer for the Sentiment column
    label_indexer = StringIndexer(inputCol="Sentiment", outputCol="Label2")
    label_indexer_model = label_indexer.fit(spark_df)
    spark_df = label_indexer_model.transform(spark_df)

    # Label mappings
    label_mappings = {i: label for i, label in enumerate(label_indexer_model.labels)}

    # Drop rows with empty 'Tweet-Comment' column
    spark_df = spark_df.dropna(subset=['Tweet-Comment'])

    # Tokenizer, StopWordsRemover, CountVectorizer
    tokenizer = Tokenizer(inputCol="Tweet-Comment", outputCol="tokens")
    stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", stopWords=stop_words)
    count_vectorizer = CountVectorizer(inputCol="filtered_tokens", outputCol="features", vocabSize=10000, minDF=5)

    # Naive Bayes Classifier
    nb = NaiveBayes(labelCol="Label2", featuresCol="features")

    # Pipeline
    pipeline = Pipeline(stages=[tokenizer, stopwords_remover, count_vectorizer, nb])

    # Fit the pipeline
    model = pipeline.fit(spark_df)

    # Accuracy Evaluation
    evaluator = MulticlassClassificationEvaluator(labelCol="Label2", predictionCol="prediction", metricName="accuracy")
    processed_data = model.transform(spark_df)
    accuracy = evaluator.evaluate(processed_data)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Classify new text
    new_text_df = spark.createDataFrame([Row(**{'Tweet-Comment': text})])

    # Transform the new text
    predictions = model.transform(new_text_df)
    result = predictions.select("Tweet-Comment", "prediction").collect()

    # Map predicted label back to the original label
    predicted_label = label_mappings[int(result[0]['prediction'])]

    return predicted_label
