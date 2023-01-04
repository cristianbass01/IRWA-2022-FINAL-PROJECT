import nltk

from myapp.search.objects import Document

nltk.download('stopwords')
from collections import defaultdict
from array import array
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math
import numpy as np
import collections
from numpy import linalg as la
import re
import json


def build_terms(line):
    """
        Preprocess the tweet text removing stop words, stemming,
        transforming in lowercase and return the tokens of the text.

        Argument:
        line -- string (text) to be preprocessed

        Returns:
        line - a list of tokens corresponding to the input text after the preprocessing
        """

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    # START CODE
    line = line.lower()  # Transform in lowercase
    line = line.split()  # Tokenize the text to get a list of terms
    tweet_text = []
    for word in line:
        # let's try to maintain the links in the correct format for the last part
        if "https" not in word:  # we maintain the # and @ because have relevance and we delete all the punctuation
            word = re.sub(r'[^\w\s#@]', '', word)
            word = re.sub(r'_', '', word)

        if word:
            tweet_text.append(word)

    line = [word for word in tweet_text if
            not word in stop_words]  # eliminate the stopwords (HINT: use List Comprehension)
    line = [stemmer.stem(word) for word in line]  # perform stemming (HINT: use List Comprehension)
    # END CODE
    return line


def create_tfidf_index(corpus):
    """
        Implement the inverted index and compute tf, df and idf

        Argument:
        lines -- collection of Wikipedia articles
        num_documents -- total number of documents

        Returns:
        index - the inverted index (implemented through a Python dictionary) containing terms as keys and the corresponding
        list of document these keys appears in (and the positions) as values.
        tf - normalized term frequency for each term in each document
        df - number of documents each term appear in
        idf - inverse document frequency of each term
        """

    index = defaultdict(list)
    tf = defaultdict(list)  # term frequencies of terms in documents (documents in the same order as in the main index)
    df = defaultdict(int)  # document frequencies of terms in the input document
    idf = defaultdict(float)
    num_documents = len(corpus.keys())

    for doc_id in corpus.keys():
        tweet = corpus[doc_id]
        terms = tweet.preprocess

        ## ===============================================================
        ## create the index for the **current page** and store it in current_page_index
        ## current_doc_index ==> { ‘term1’: [current_doc, [list of positions]], ...,‘term_n’: [current_doc, [list of positions]]}

        ## current_page_index ==> { ‘web’: [1, [0]], ‘retrieval’: [1, [1,4]], ‘information’: [1, [2]]}

        ## the term ‘web’ appears in document 1 in positions 0,
        ## the term ‘retrieval’ appears in document 1 in positions 1 and 4
        ## ===============================================================

        current_doc_index = {}

        for position, term in enumerate(terms):  ## terms contains tweet text
            try:
                # if the term is already in the dict append the position to the corresponding list
                current_doc_index[term][1].append(position)
            except:
                # Add the new term as dict key and initialize the array of positions and add the position
                current_doc_index[term] = [doc_id, array('I', [position])]  # 'I' indicates unsigned int (int in Python)

        # normalize term frequencies
        # Compute the denominator to normalize term frequencies (formula 2 above)
        # norm is the same for all terms of a document.
        norm = 0
        for term, posting in current_doc_index.items():
            # posting will contain the list of positions for current term in current document.
            # posting ==> [current_doc, [list of positions]]
            # you can use it to infer the frequency of current term.
            norm += len(posting[1]) ** 2
        norm = math.sqrt(norm)

        # calculate the tf(dividing the term frequency by the above computed norm) and df weights
        for term, posting in current_doc_index.items():
            # append the tf for current term (tf = term frequency in current doc/norm)
            tf[term].append(np.round(len(posting[1]) / norm, 4))  ## SEE formula (1) above
            # increment the document frequency of current term (number of documents containing the current term)
            df[term] += 1  # increment DF for current term

        # merge the current doc index with the main index
        for term_doc, posting_doc in current_doc_index.items():
            index[term_doc].append(posting_doc)

        # Compute IDF following the formula (3) above. HINT: use np.log
        for term in df:
            idf[term] = np.round(np.log(float(num_documents / df[term])), 4)

    return index, tf, df, idf


def search_in_corpus(corpus, query, index, idf, tf, function='tf_idf', k1=1.4, b=0.75):
    """
    output is the list of documents that contain any of the query terms.
    So, we will get the list of documents for each query term, and take the union of them.
    """
    query = build_terms(query)
    docs = set()
    for term in query:
        try:
            # store in term_docs the ids of the docs that contain "term"
            term_docs = [posting[0] for posting in index[term]]

            # docs = docs Union term_docs
            docs |= set(term_docs)
        except:
            # term is not in index
            pass
    docs = list(docs)

    # if function == 'tf_idf':
    doc_scores =  rankTweetsPersonalized(corpus, query, docs, index, idf, tf)
    """elif function == 'personalized':
        ranked_docs, doc_scores = rankTweetsPersonalized(query, docs, index, idf, tf)
    elif function == 'bm25':
        ranked_docs, doc_scores = bm25_score(query, docs, index, idf, tf, k1, b)
    else:
        raise Exception('Error: no function of type', function)
    """

    return doc_scores


def rank_documents(terms, docs, index, idf, tf):
    """
    Perform the ranking of the results of a search based on the tf-idf weights

    Argument:
    terms -- list of query terms
    docs -- list of documents, to rank, matching the query
    index -- inverted index data structure
    idf -- inverted document frequencies
    tf -- term frequencies


    Returns:
    Print the list of ranked documents
    """

    # I'm interested only on the element of the docVector corresponding to the query terms
    # The remaining elements would became 0 when multiplied to the query_vector
    doc_vectors = defaultdict(lambda: [0] * len(
        terms))  # I call doc_vectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    query_vector = [0] * len(terms)

    # compute the norm for the query tf
    query_terms_count = collections.Counter(terms)  # get the frequency of each term in the query.
    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):  # termIndex is the index of the term in the query
        if term not in index:
            continue

        # Compute tf*idf(normalize TF as done with documents)
        query_vector[termIndex] = query_terms_count[term] / query_norm * idf[term]

        # Generate doc_vectors for matching docs
        for doc_index, (doc, postings) in enumerate(index[term]):
            # tf[term][0] will contain the tf of the term "term" in the docs
            if doc in docs:
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]

    # Calculate the score of each doc
    # compute the cosine similarity between queryVector and each docVector:
    doc_scores = [[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items()]
    doc_scores.sort(reverse=True)

    if len(doc_scores) == 0:
        print("No results found, try again")
    return doc_scores


def rankTweetsPersonalized(corpus, terms, docs, index, idf, tf):
    """
    Perform the ranking of the results of a search based on the tf-idf weights

    Argument:
    terms -- list of query terms
    docs -- list of tweet ids, to rank, matching the query
    index -- inverted index data structure
    idf -- inverted document frequencies
    tf -- term frequencies
    titleIndex -- mapping between page id and page title

    Returns:
    resultScores --  List of ranked scores of tweets
    """
    # I'm interested only on the element of the docVector corresponding to the query terms
    # The remaining elements would became 0 when multiplied to the query_vector
    doc_vectors = defaultdict(lambda: [0] * len(
        terms))  # I call doc_vectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    query_vector = [0] * len(terms)

    # compute the norm for the query tf
    query_terms_count = collections.Counter(terms)  # get the frequency of each term in the query.
    # Example: collections.Counter(["hello","hello","world"]) --> Counter({'hello': 2, 'world': 1})
    # HINT: use when computing tf for query_vector

    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):  # termIndex is the index of the term in the query
        if term not in index:
            continue

        ## Compute tf*idf(normalize TF as done with documents)
        query_vector[termIndex] = query_terms_count[term] / query_norm * idf[term]

        # Generate doc_vectors for matching docs
        for doc_index, (doc, postings) in enumerate(index[term]):
            # tf[term][0] will contain the tf of the term "term" in the doc 26
            if doc in docs:
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]

    # Calculate the score of each doc
    # compute the cosine similarity between queryVector and each docVector:
    doc_scores = [[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items()]

    # Once we have the scoring of TF-IDF apply variations to the ranking
    # Get all the tweet data from the docs
    query_tweets = {}
    likes_count = []
    likesByFollow = []
    retweets_count = []
    retweetsByFollow = []

    for k in docs:
        tweet = corpus[k]
        query_tweets[k] = tweet
        likes_count.append(tweet.likes)
        retweets_count.append(tweet.retweets)
        likesByFollow.append(tweet.likes / (tweet.followers+1) )
        retweetsByFollow.append(tweet.retweets / (tweet.followers+1))

    # Normalize the likes and retweets among all the query output tweets
    likes_norm = la.norm(likes_count)
    ret_norm = la.norm(retweets_count)
    likes_norma = [float(r / likes_norm) for r in likes_count]
    retweets_norma = [float(r / ret_norm) for r in retweets_count]

    # Take care of the hsahtags freq

    # Calculate the ponderation of the hashtags, we want to undervaluate tweets that add to much hashtags for spam
    # To do so we calculate 1/num of hashtags to add to the score
    # Also if it has a trend hashtag need to increase its score
    # Select every stat weight in the popularity score

    likes = 0.40
    rets = 0.40
    l_f = 0.10
    r_f = 0.010

    pop_scores = {}
    list_ids = list(docs)
    for x in range(len(list_ids)):
        pop_scores[list_ids[x]] = likes * likes_norma[x] + rets * retweets_norma[x] + l_f * likesByFollow[x] + r_f * \
                                  retweetsByFollow[x]

    # select the value of tfid and popularity scores in the final score
    tf_idfs = 0.3
    pops = 0.7

    tfidf_norm = la.norm([r[0] for r in doc_scores])
    pops_norm = la.norm(list(pop_scores.values()))

    # normalize to reduce the difference in scoring in both methods
    tweetScores = [
        [np.dot(curTweetVec, query_vector) / tfidf_norm * tf_idfs + pops * pop_scores[tweet_id] / pops_norm, tweet_id]
        for tweet_id, curTweetVec in doc_vectors.items()]

    tweetScores.sort(reverse=True)

    if len(tweetScores) == 0:
        print("No results found, try again")

    # return rank punctuation and ids
    return tweetScores


def hashtagsFreq(tweets):
    hashtags = []
    for t in tweets.values():
        for hashtag in t["entities"]["hashtags"]:
            hashtags.append("#"+hashtag["text"])
    hash_count = collections.Counter(hashtags)
    hash_norm = la.norm(list(hash_count.values()))
    for h, c in hash_count.items():
        hash_count[h] = c/hash_norm
    return hash_count