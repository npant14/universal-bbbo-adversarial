import pandas as pd
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

def get_data(filename):
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    df = pd.read_csv(filename)
    words = basic_clean_ngram(''.join(str(df['text'].tolist())))

    bigrams = pd.Series(nltk.ngrams(words, 2)).value_counts()
    return bigrams

def get_next_words(bigrams_series):
    word_pairs = {}

    bigrams_series.sort_values()
    for row in bigrams_series.items():
        print(row)
        if row[0][0] in word_pairs:
            word_pairs[row[0][0]].append(row[0][1])
        else:
            word_pairs[row[0][0]] = [row[0][1]]


    #print(word_pairs)
    return word_pairs

def basic_clean(text):
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    text = (unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore').lower())

    words = re.sub(r'[^\w\s]', '', text).split()
    return words
    #return [word for word in words if word not in stopwords]
    #return [wnl.lemmatize(word) for word in words if word not in stopwords]

def basic_clean_ngram(text):
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    text = (unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore').lower())

    words = re.sub(r'[^\w\s]', '', text).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]


# tb = get_data("tweets.csv")

# get_next_words(tb)
