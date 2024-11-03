from django.shortcuts import render

from pymongo import MongoClient
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pymongo import MongoClient
import re
import nltk


nltk.download('punkt')
nltk.download('stopwords')



#------------------------------------------------------------------------------------------------------------------

# Connect to MongoDB
mongo_client = MongoClient('mongodb://localhost:27017/')  # Update with your MongoDB connection string
db = mongo_client['twitter_sentiment']  # Database name
collection = db['predictions']  # Collection name

#---------------------------------------------------------------------------



def classify(request) :
    
   error = False
   error_text = ""
   prediction = ""
   text = ""

   if request.method == 'POST':

      text = request.POST.get('text')
      print(len(text.strip()))
      if len(text.strip()) > 0 :
         error = False
         from .ml import classify_text
         prediction = classify_text(text)
      
      else : 
         error = True
         error_text = "the Text is empty!! PLZ Enter Your Text."

   context = {
      "error" : error,
      "error_text" : error_text,
      "prediction" : prediction,
      "text" : text,
      "text_len" : len(text.strip())
   }

   print(context)
   return render(request, 'classify.html', context)




def home(request):
   return render(request, 'home.html')








from django.shortcuts import render
import pandas as pd
from pymongo import MongoClient
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

# MongoDB connection
mongo_client = MongoClient('mongodb://localhost:27017/')  # Update with your MongoDB connection string
db = mongo_client['twitter_sentiment']  # Database name
collection = db['predictions']  # Collection name

def dashboard(request):
    # Fetch data from MongoDB
    data = collection.find()
    df = pd.DataFrame(list(data))
    
    # Handle missing data
    df.dropna(inplace=True)  # Drop rows with missing values

    # Sentiment analysis using spaCy
    nlp = spacy.load('en_core_web_sm')
    text_data = ' '.join(df['Tweet-Comment'])  # Use the actual text column
    doc = nlp(text_data)
    
    # Named Entity Recognition (NER)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Word cloud visualization
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

    # Plot sentiment frequencies
    sentiment_counts = df['Sentiment'].value_counts()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title('Sentiment Frequency')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig('sentiment_frequency.png')
    plt.close()

    # Dendrogram
    # For clustering, you may need to convert sentiments to numeric values
    sentiment_mapping = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
    df['Sentiment_Numeric'] = df['Sentiment'].map(sentiment_mapping)
    linked = linkage(df[['Sentiment_Numeric']], method='ward')

    plt.figure(figsize=(10, 5))
    dendrogram(linked)
    plt.title('Dendrogram of Sentiments')
    plt.savefig('dendrogram.png')
    plt.close()

    # Generate the WordCloud image
    wordcloud_image_path = 'wordcloud.png'
    wordcloud.to_file(wordcloud_image_path)

    context = {
        'sentiment_plot': 'sentiment_frequency.png',
        'dendrogram_plot': 'dendrogram.png',
        'wordcloud_plot': wordcloud_image_path,
        'entities': entities
    }
    
    return render(request, 'dashboard.html', context)
