
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import findspark
findspark.init()
from nltk.corpus import stopwords
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, pandas_udf,col, lower, regexp_replace
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, ArrayType
from pyspark.ml.feature import CountVectorizer, StringIndexer, Tokenizer, StopWordsRemover
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from sklearn.metrics import confusion_matrix
from pyspark.ml import PipelineModel, Pipeline, Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable


# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Define English stopwords
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

# Rename columns
for i, col_name in enumerate(columns):
    data = data.withColumnRenamed(f'_c{i}', col_name)
    validation = validation.withColumnRenamed(f'_c{i}', col_name)

# Define the StringIndexer for the label column (index the labels)
label_indexer = StringIndexer(inputCol="Label", outputCol="Label2")

# Drop rows with empty 'Text' column
data = data.dropna(subset=['Text'])
validation = validation.dropna(subset=['Text'])

# Fit StringIndexer on data
# Define the StringIndexer for the label column (index the labels)
label_indexer = StringIndexer(inputCol="Label", outputCol="Label2")

# # Define your index mapping
# class_index_mapping = { "Negative": 0, "Positive": 1, "Neutral": 2, "Irrelevant": 3 }

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
    # Remove links starting with https://, http://, www., or containing .com
    df = df.withColumn(outputCol, regexp_replace(df[inputCol], r'https?://\S+|www\.\S+|S+\.com\S+|youtu\.be/\S+', ''))
    # Remove words starting with # or @
    df = df.withColumn(outputCol, regexp_replace(df[outputCol], r'(@|#)\w+', ''))
    # Convert text to lowercase
    df = df.withColumn(outputCol, lower(df[outputCol]))
    # Remove non-alpha characters
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
# # Assuming you have a list of labels like this
# label_mapping_list = ["Negative", "Positive", "Neutral", "Irrelevant"]

# # Create a dictionary mapping from the label indices to the actual labels
# label_mapping = {index: label for index, label in enumerate(label_mapping_list)}


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

    # Rename columns
    for i, col_name in enumerate(columns):
        data = data.withColumnRenamed(f'_c{i}', col_name)
        validation = validation.withColumnRenamed(f'_c{i}', col_name)

    # Define the StringIndexer for the label column (index the labels)
    label_indexer = StringIndexer(inputCol="Label", outputCol="Label2")

    # Drop rows with empty 'Text' column
    data = data.dropna(subset=['Text'])
    validation = validation.dropna(subset=['Text'])

    # Fit StringIndexer on data
    # Define the StringIndexer for the label column (index the labels)
    label_indexer = StringIndexer(inputCol="Label", outputCol="Label2")

    # # Define your index mapping
    # class_index_mapping = { "Negative": 0, "Positive": 1, "Neutral": 2, "Irrelevant": 3 }

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
    # Create a DataFrame for the new text
    new_text_df = spark.createDataFrame([('company1', text)], ["Company", "Text"])  # Use None for 'Company' as it's not used in prediction

    # Clean the new text data
    cleaned_new_text = clean_new_text(new_text_df, inputCol="Text", outputCol="Text")

    # Use the trained model to make predictions on the cleaned new text
    predictions = model.transform(cleaned_new_text)

    # Select the relevant output (original text and prediction)
    result = predictions.select("Text", "prediction").collect()

    sentiment_label = label_mappings.get(result[0]['prediction'], "Unknown")
    
    # Return the result
    if result:  # Check if there's any result
        # Get the prediction index from the first result
        prediction_index = result[0]['prediction']  # Accessing the prediction
        # Map to actual sentiment label using the dictionary
        sentiment_label = label_mappings.get(prediction_index, "Unknown")
        return sentiment_label
    else:
        return "No prediction available"

# Call the classify_text function with sample inputs


# # Iterate through the sample texts and classify each
# for text in sample_texts:
#     prediction = classify_text(text)
#     print(f"Text: '{text}' | Prediction: {prediction[0]['prediction']}")



from pyspark.ml.classification import NaiveBayes






from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

def classify_text_nb(text: str):
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

    # Rename columns
    for i, col_name in enumerate(columns):
        data = data.withColumnRenamed(f'_c{i}', col_name)
        validation = validation.withColumnRenamed(f'_c{i}', col_name)

    # StringIndexer for the label column
    label_indexer = StringIndexer(inputCol="Label", outputCol="Label2")
    label_indexer_model = label_indexer.fit(data)
    data = label_indexer_model.transform(data)
    validation = label_indexer_model.transform(validation)

    # Label mappings
    label_mappings = {i: label for i, label in enumerate(label_indexer_model.labels)}

    # Drop rows with empty 'Text' column
    data = data.dropna(subset=['Text'])
    validation = validation.dropna(subset=['Text'])

    # Tokenizer, StopWordsRemover, CountVectorizer
    tokenizer = Tokenizer(inputCol="Text", outputCol="tokens")
    stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", stopWords=stop_words)
    count_vectorizer = CountVectorizer(inputCol="filtered_tokens", outputCol="features", vocabSize=10000, minDF=5)

    # Naive Bayes Classifier
    nb = NaiveBayes(labelCol="Label2", featuresCol="features")

    # Pipeline
    pipeline = Pipeline(stages=[tokenizer, stopwords_remover, count_vectorizer, nb])

    # Fit the pipeline
    model = pipeline.fit(data)

    # Accuracy Evaluation
    evaluator = MulticlassClassificationEvaluator(labelCol="Label2", predictionCol="prediction", metricName="accuracy")
    processed_data = model.transform(data)
    accuracy = evaluator.evaluate(processed_data)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Classify new text
    new_text_df = spark.createDataFrame([('company1', text)], ["Company", "Text"])

    cleaned_new_text = clean_new_text(new_text_df, inputCol="Text", outputCol="Text")
    predictions = model.transform(cleaned_new_text)
    result = predictions.select("Text", "prediction").collect()
    i = result[0]['prediction']
    return label_mappings[int(i)]
