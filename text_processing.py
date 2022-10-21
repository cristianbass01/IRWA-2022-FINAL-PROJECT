# importing library nltk
import datetime
import time

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import regex as re
import numpy as np
import json

nltk.download('stopwords')


def load_data(path_tweets, path_docs_tweet):
    id_tweet = {}
    doc_tweet = {}
    with open(path_tweets) as tp:
        for line in tp.readlines():
            tweet = json.loads(line)
            id_tweet[tweet['id']] = tweet

    with open(path_docs_tweet) as dp:
        for line in dp.readlines():
            line = line.split()
            doc_tweet[line[0]] = id_tweet[int(line[1])]
    return doc_tweet


def get_text(tweet):
    try:
        return tweet['full_text']
    except KeyError:
        return ' '


def get_username(tweet):
    try:
        return tweet['user']['screen_name']
    except KeyError:
        return ' '


def get_date(tweet):
    try:
        created_at = datetime.datetime.strptime(tweet['created_at'], "%a %b %d %X %z %Y" )
        return created_at.strftime('%A %d %B %Y')
    except KeyError:
        return ' '


def get_hashtags(tweet):
    try:
        hashtags = []
        for hash in  tweet['entities']['hashtags']:
                hashtags.append('##' + hash['text'])
        return ' '.join(hashtags)
    except KeyError:
        return ' '


def get_likes(tweet):
    try:
        return str(tweet['favorite_count'])
    except KeyError:
        return ' '


def get_retweets(tweet):
    try:
        return str(tweet['retweet_count'])
    except KeyError:
        return ' '


def get_url(tweet):
    try:
        return 'https://twitter.com/' + tweet['user']['screen_name'] + '/status/'+ str(tweet['id'])
    except KeyError:
        return ' '


def build_map(dict_docs_tweet):
    doc_map = {}
    for doc in dict_docs_tweet.keys():
        tweet = dict_docs_tweet[doc]
        items_list = [get_text(tweet), get_username(tweet), get_date(tweet), get_hashtags(tweet), get_likes(tweet), get_retweets(tweet), get_url(tweet)]
        doc_map[doc] = " | ".join(items_list)
    return doc_map


def preprocess(str_line):
    """
    Argument:
    line -- string (text) to be preprocessed

    Returns:
    line - a list of tokens corresponding to the input text after the preprocessing
    """

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    str_line = str_line.lower()
    str_line = re.sub(r'(\s)(##)[^\s]+', ' ', str_line) #Removing ## hashtags only
    str_line = re.sub(r'(\s)(http)[^\s]+', ' ', str_line) # Removing links
    str_line = re.sub(r'[^\w\s#@]+', ' ', str_line) # Removing punctuation marks
    str_line = str_line.split()  # Tokenize the text to get a list of terms
    str_line = [x for x in str_line if x not in stop_words]  # Eliminate the stopwords
    str_line = [stemmer.stem(word) for word in str_line]  # Perform stemming
    return str_line



TWEETS_PATH = 'data/tw_hurricane_data.json'
DOCS_PATH = 'data/tweet_document_ids_map.csv'
doc_to_tweet = load_data(TWEETS_PATH, DOCS_PATH)
print("Total number of docs of tweets: {}".format(len(doc_to_tweet)))

docs_map = build_map(doc_to_tweet)

for index in range(2):
    doc = list(docs_map.keys())[index]
    print("Original {} text line:\n    {} \n".format(doc, docs_map[doc]))


def build_prep_map(dict_docs_tweet):
    prep_doc_map = {}
    for doc in dict_docs_tweet.keys():
        tweet = dict_docs_tweet[doc]
        prep_doc_map[doc] = preprocess(get_text(tweet)) + preprocess(get_username(tweet)) + preprocess(get_date(tweet))
    return prep_doc_map



start_time = time.time()
prep_docs_map = build_prep_map(doc_to_tweet)
print("Total time to preprocess tweets: {} seconds" .format(np.round(time.time() - start_time, 2)))

for index in range(2):
    doc = list(prep_docs_map.keys())[index]
    print("Preprocess {} text line:\n   {}\n".format(doc, prep_docs_map[doc]))

