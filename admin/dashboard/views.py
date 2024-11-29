import base64
from io import BytesIO
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
         from .ml import classify_text_lr, classify_text_nb
         prediction = None
         print(request.POST.get("drop"))
               
         if request.POST.get("drop") == "linear_regression":
             prediction = classify_text_lr(text)
         elif request.POST.get('drop') == 'naive_bayes':
             print('hi')
             prediction = classify_text_nb(text)
      
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
   return render(request, 'classify2.html', context)




def home(request):
   return render(request, 'home.html')



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from wordcloud import WordCloud
from collections import Counter
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from io import BytesIO
import base64
from pymongo import MongoClient
from django.shortcuts import render


def generate_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_png = buf.getvalue()
    buf.close()
    return base64.b64encode(image_png).decode('utf-8')



def dashboard(request):
    # Fetch data from MongoDB and load into DataFrame
    data = list(collection.find())
    df = pd.DataFrame(data)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Convert sentiments to numeric
    df['Sentiment_Numeric'] = df['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
    df['Predicted_Sentiment_Numeric'] = df['Predicted_Sentiment'].map({'Positive': 1, 'Negative': 0, 'Neutral': 2, 'Irrelevant': 3})

    # Handle missing or non-finite values
    df['Sentiment_Numeric'] = df['Sentiment_Numeric'].fillna(0)
    df['Predicted_Sentiment_Numeric'] = df['Predicted_Sentiment_Numeric'].fillna(0)

    # Prepare comments and representative words
    comments = df['Tweet-Comment'].fillna('').tolist()
    representative_words = [comment.split()[0] if comment else '' for comment in comments]

    # 1. Dendrogram
    fig, ax = plt.subplots()
    linked = linkage(df[['Sentiment_Numeric', 'Predicted_Sentiment_Numeric']], method='ward')
    dendrogram(linked, ax=ax, labels=representative_words, leaf_rotation=90, leaf_font_size=10)
    dendrogram_img = generate_image(fig)

    # 2. Word Cloud
    text_data = ' '.join(comments)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    wordcloud_img = generate_image(fig)

    # 3. Word Frequencies per Class
    fig, ax = plt.subplots()
    sentiment_counts = df['Sentiment'].value_counts()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax)
    ax.set_title("Word Frequencies per Sentiment Class")
    word_freq_img = generate_image(fig)

    # 4. Named Entity Recognition (NER)
    nlp = spacy.load("en_core_web_sm")
    entities = []
    for tweet in comments:
        doc = nlp(tweet)
        entities.extend([(ent.text, ent.label_) for ent in doc.ents])
    entity_counter = Counter([label for _, label in entities])
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=list(entity_counter.keys()), y=list(entity_counter.values()), ax=ax)
    ax.set_title("Named Entity Recognition (NER)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    ner_img = generate_image(fig)

    # 5. Bigram Analysis
    vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')
    bigrams = vectorizer.fit_transform(comments)
    bigram_counts = pd.DataFrame(bigrams.toarray(), columns=vectorizer.get_feature_names_out()).sum(axis=0).sort_values(ascending=False)
    fig, ax = plt.subplots()
    bigram_counts.head(10).plot(kind='bar', ax=ax)
    ax.set_title("Top 10 Bigrams")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    bigram_img = generate_image(fig)

    # 6. Top Words
    vectorizer = CountVectorizer(stop_words='english')
    words = vectorizer.fit_transform(comments)
    word_counts = pd.DataFrame(words.toarray(), columns=vectorizer.get_feature_names_out()).sum(axis=0).sort_values(ascending=False)
    fig, ax = plt.subplots()
    word_counts.head(10).plot(kind='bar', ax=ax)
    ax.set_title("Top 10 Words")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    top_words_img = generate_image(fig)

    # 7. Comment Length Distribution Based on Labels
    df['Comment_Length'] = df['Tweet-Comment'].str.len()
    fig, ax = plt.subplots()
    sns.boxplot(x='Sentiment', y='Comment_Length', data=df, ax=ax)
    ax.set_title("Distribution of Comment Lengths Based on Labels")
    comment_length_img = generate_image(fig)

    # 8. Sentiment Trends Over Time
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        sentiment_trends = df.groupby([df['Date'].dt.date, 'Sentiment']).size().unstack(fill_value=0)
        fig, ax = plt.subplots()
        sentiment_trends.plot(ax=ax, kind='line', marker='o')
        ax.set_title("Sentiment Trends Over Time")
        sentiment_trends_img = generate_image(fig)
    else:
        sentiment_trends_img = None

    # 9. Sentiment Ratio
    fig, ax = plt.subplots()
    sentiment_counts.plot.pie(autopct='%1.1f%%', ax=ax)
    ax.set_ylabel('')
    ax.set_title("Sentiment Ratio")
    sentiment_ratio_img = generate_image(fig)

    # 10. Word Count Distribution
    df['Word_Count'] = df['Tweet-Comment'].apply(lambda x: len(x.split()))
    fig, ax = plt.subplots()
    sns.histplot(df['Word_Count'], kde=True, bins=20, ax=ax)
    ax.set_title("Word Count Distribution")
    word_count_img = generate_image(fig)

    # Add all images to context
    context = {
        'dendrogram_img': dendrogram_img,
        'wordcloud_img': wordcloud_img,
        'word_freq_img': word_freq_img,
        'ner_img': ner_img,
        'bigram_img': bigram_img,
        'top_words_img': top_words_img,
        'comment_length_img': comment_length_img,
        'sentiment_trends_img': sentiment_trends_img,
        'sentiment_ratio_img': sentiment_ratio_img,
        'word_count_img': word_count_img,
    }

    return render(request, 'dashboard.html', context)








# from django.shortcuts import render
# import pandas as pd
# from pymongo import MongoClient
# import spacy
# import matplotlib.pyplot as plt
# import seaborn as sns
# from wordcloud import WordCloud
# import numpy as np
# from scipy.cluster.hierarchy import dendrogram, linkage

# # MongoDB connection
# mongo_client = MongoClient('mongodb://localhost:27017/')  # Update with your MongoDB connection string
# db = mongo_client['twitter_sentiment']  # Database name
# collection = db['predictions']  # Collection name

# def generate_image(fig):
#     buf = BytesIO()
#     fig.savefig(buf, format='png')
#     buf.seek(0)
#     image_png = buf.getvalue()
#     buf.close()
#     return base64.b64encode(image_png).decode('utf-8')

# def dashboard(request):
#     # Fetch data from MongoDB and load into DataFrame
#     data = list(collection.find())
#     df = pd.DataFrame(data)

#     # Strip whitespace from column names
#     df.columns = df.columns.str.strip()

#     # Convert sentiments to numeric
#     df['Sentiment_Numeric'] = df['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
#     df['Predicted_Sentiment_Numeric'] = df['Predicted_Sentiment'].map({'Positive': 1, 'Negative': 0, 'Neutral': 2, 'Irrelevant': 3})

#     # Check for NaN values in numeric columns
#     if df['Sentiment_Numeric'].isnull().any() or df['Predicted_Sentiment_Numeric'].isnull().any():
#         print("NaN values found in Sentiment_Numeric or Predicted_Sentiment_Numeric")
        
#         # Handle NaN values by filling them with a default value, e.g., 0
#         df['Sentiment_Numeric'] = df['Sentiment_Numeric'].fillna(0)
#         df['Predicted_Sentiment_Numeric'] = df['Predicted_Sentiment_Numeric'].fillna(0)

#     # Check again for non-finite values
#     if not np.isfinite(df[['Sentiment_Numeric', 'Predicted_Sentiment_Numeric']]).all().all():
#         print("Non-finite values found in the numeric columns.")
#         context = {
#             'dendrogram_img': None,
#             'wordcloud_img': None,
#             'word_freq_img': None,
#             'ner_img': None,
#         }
#         return render(request, 'dashboard.html', context)
    
#     comments = df['Tweet-Comment'].tolist()
    
#     # Create a list of representative words (e.g., the first word of each comment)
#     representative_words = [comment.split()[0] if comment else '' for comment in comments]
    
#     # 1. Dendrogram
#     fig, ax = plt.subplots()
#     linked = linkage(df[['Sentiment_Numeric', 'Predicted_Sentiment_Numeric']], method='ward')
#     dendrogram(linked, ax=ax, labels=representative_words, leaf_rotation=90, leaf_font_size=10)
#     dendrogram_img = generate_image(fig)

#     # 2. Word Cloud
#     text_data = ' '.join(df['Tweet-Comment'].fillna(''))
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
#     fig, ax = plt.subplots()
#     ax.imshow(wordcloud, interpolation='bilinear')
#     ax.axis('off')
#     wordcloud_img = generate_image(fig)

#     # 3. Plot Word Frequencies per Class
#     fig, ax = plt.subplots()
#     sentiment_counts = df['Sentiment'].value_counts()
#     sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax)
#     ax.set_title("Word Frequencies per Sentiment Class")
#     word_freq_img = generate_image(fig)

#     # 4. Named Entity Recognition (NER) visualization
#     nlp = spacy.load("en_core_web_sm")

# # Extract entities from text
#     entities = []
#     for tweet in df['Tweet-Comment'].fillna(''):
#         doc = nlp(tweet)
#         entities.extend([(ent.text, ent.label_) for ent in doc.ents])

#     # Count the entities
#     entity_counter = Counter([label for _, label in entities])

#     # Plot the entities
#     fig, ax = plt.subplots(figsize=(12, 6))  # Increase figure size
#     sns.barplot(x=list(entity_counter.keys()), y=list(entity_counter.values()), ax=ax)
#     ax.set_title("Named Entity Recognition (NER)")
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate and align labels
#     plt.tight_layout()  # Adjust layout to prevent clipping

#     # Save or display the image
#     ner_img = generate_image(fig)

#     # Add all images to context
#     context = {
#         'dendrogram_img': dendrogram_img,
#         'wordcloud_img': wordcloud_img,
#         'word_freq_img': word_freq_img,
#         'ner_img': ner_img,
#     }

#     return render(request, 'dashboard.html', context)