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
