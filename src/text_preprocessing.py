
# Step 1: Import Required NLP Libraries
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Step 2: Define Text Cleaning Function
def preprocess_text(text):
    # Lowerxase
    text = text.lower()

    # Remove URLs
    text  = re.sub(r'http\S+|www.\S+','',text)

    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]','',text)

    # Remove stopwards and apply stemming
    words = text.split()
    cleaned_words = [stemmer.stem(word) for word in words if word not in stopwords]

    return ' '.join(cleaned_words)

import pandas as pd
data=pd.read_csv('C:\\CodeShell_Core\\GitHub_Repository\\fake-news-detector\\notebooks\\combined_news.csv')

# Step 3: Apply Preprocessing to the Dataset
data['clean_text'] = data['text'].apply(preprocess_text)

# Step 4: Save Preprocessed Data
data.to_csv('C:\\CodeShell_Core\\GitHub_Repository\\fake-news-detector\\src\\cleaned_news',index=False)

