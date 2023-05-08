import pandas as pd
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import torch

def get_data(filename):
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    df = pd.read_csv(filename)
    words = basic_clean_ngram(''.join(str(df['description'].tolist()[0])))
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

    return word_pairs

def basic_clean(text):
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    text = (unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore').lower())

    words = re.sub(r'[^\w\s]', '', text).split()
    return words

def basic_clean_ngram(text):
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    text = (unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore').lower())

    words = re.sub(r'[^\w\s]', '', text).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]


def tokenize_word(word, token_dict, untoken_dict):
    if word not in token_dict:
        token = len(token_dict) + 2 ## put in a +2 here to account for start and stop tokens
        token_dict[word] = token
        untoken_dict[token] = word
    return token_dict[word]

def tokenizing_sst2(sentence, token_dict, untoken_dict):
    input_ids = []
    attention_mask = []
    ## basic_clean removes punctuation and casing
    sentence = basic_clean(sentence)

    ## 0 is start token
    tokenized_sentence = [0]
    
    for word in sentence:
        token = tokenize_word(word, token_dict, untoken_dict)
        tokenized_sentence.append(token)

    ## 1 is stop token
    tokenized_sentence.append(1)

    ## pad sentence with 0s to len 512
    tokenized_sentence  = tokenized_sentence +  [0] * (512 - len(tokenized_sentence))
    ## create attention mask
    attention_mask = [1] * (len(tokenized_sentence)) + [0] * (512 - len(tokenized_sentence))

    return torch.stack([torch.tensor(tokenized_sentence),
                        torch.tensor(attention_mask)], dim=0)

tb = get_data("books_1.Best_Books_Ever.csv")

get_next_words(tb)
