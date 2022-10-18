# importing library nltk
import nltk
from collections import defaultdict
from array import array
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math
import numpy as np
import collections
from numpy import linalg as la
import json

nltk.download('stopwords')

def load_data(path_tweets, path_docs_tweet):

    id_tweet = {}
    doc_tweet = {}
    with open(tweets_path) as tp:
        for line in tp.readlines():
            tweet = json.loads(line)
            id_tweet[tweet['id']] = tweet

    with open(docs_path) as dp:
        for line in dp.readlines():
            line = line.split()
            doc_tweet[line[0]] = id_tweet[int(line[1])]
    return doc_tweet


tweets_path = 'data/tw_hurricane_data.json'
docs_path = 'data/tweet_document_ids_map.csv'
doc_to_tweet = load_data(tweets_path, docs_path)
print("Total number of docs of tweets: {}".format(len(doc_to_tweet)))


def preprocess(str_line):
    """
    Preprocess the article text (title + body) removing stop words, stemming,
    transforming in lowercase and return the tokens of the text.

    Argument:
    line -- string (text) to be preprocessed

    Returns:
    line - a list of tokens corresponding to the input text after the preprocessing
    """

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    str_line = str_line.lower()
    str_line = str_line.split()  # Tokenize the text to get a list of terms
    str_line = [x for x in str_line if x not in stop_words]  # eliminate the stopwords
    str_line = [stemmer.stem(word) for word in str_line]  # perform stemming
    return str_line

