# importing library nltk
import string
import time

import nltk
from collections import defaultdict
from array import array
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import regex as re
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


def build_map(dict_docs_tweet):
    doc_map = {}
    for doc in dict_docs_tweet.keys():
        tweet = dict_docs_tweet[doc]
        try:
            doc_map[doc] = [tweet['full_text']]
        except KeyError:
            doc_map[doc] += ['null']

        try:
            doc_map[doc] += [tweet['user']['name']]
        except KeyError:
            doc_map[doc] += ['null']

        try:
            doc_map[doc] += [tweet['created_at']]
        except KeyError:
            doc_map[doc] += ['null']

        try:
            doc_map[doc] += [tweet['entities']['hashtags'][0]['text']]
        except KeyError:
            doc_map[doc] += ['null']

        try:
            doc_map[doc] += [str(tweet['favorite_count'])]
        except KeyError:
            doc_map[doc] += ['null']

        try:
            doc_map[doc] += [str(tweet['retweet_count'])]
        except KeyError:
            doc_map[doc] += ['null']

        try:
            doc_map[doc] += [tweet['entities']['media'][0]['url']]
        except KeyError:
            doc_map[doc] += ['null']

        doc_map[doc] = " | ".join(doc_map[doc])
    return doc_map


def preprocess(str_line):
    """
    Preprocess the text removing stop words, stemming,
    transforming in lowercase and return the tokens of the text.

    Argument:
    line -- string (text) to be preprocessed

    Returns:
    line - a list of tokens corresponding to the input text after the preprocessing
    """

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    str_line = str_line.lower()
    str_line = re.sub('[^\w\s#@]+', ' ', str_line)
    str_line = str_line.replace_all('http', '')
    #str_line = str_line.translate(str.maketrans(dict.fromkeys("!\"$%&'()*+,-./:;<=>?[\]^_`{|}~", ' ')))
    str_line = str_line.split()  # Tokenize the text to get a list of terms
    str_line = [x for x in str_line if x not in stop_words]  # eliminate the stopwords
    str_line = [stemmer.stem(word) for word in str_line]  # perform stemming
    return str_line


start_time = time.time()
tweets_path = 'data/tw_hurricane_data.json'
docs_path = 'data/tweet_document_ids_map.csv'
doc_to_tweet = load_data(tweets_path, docs_path)
print("Total number of docs of tweets: {}".format(len(doc_to_tweet)))

docs_map = build_map(doc_to_tweet)
for index in range(5):
    print("Original doc_{} text line: {}".format(index + 1, docs_map['doc_' + str(index + 1)]))

prep_docs_map = build_map(doc_to_tweet)
for doc in prep_docs_map.keys():
    prep_docs_map[doc] = preprocess(prep_docs_map[doc])

for index in range(5):
    print("Preprocess doc_{} text line: {}".format(index + 1, prep_docs_map['doc_' + str(index + 1)]))

print("Total time to preprocess tweets: {} seconds" .format(np.round(time.time() - start_time, 2)))
