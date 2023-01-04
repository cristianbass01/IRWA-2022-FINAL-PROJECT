import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import collections
from collections import defaultdict
import math
import numpy as np
from numpy import linalg as la


def search_in_corpus(query, index, idf, tf):

    query = pre_process(query)
    tweets = set()
    for term in query:
        try:
            # store in term_docs the ids of the docs that contain "term"
            term_tweets = [tweet_id[0] for tweet_id in index[term]]
            # docs = docs Union term_docs
            tweets |= set(term_tweets)
        except:
            pass  # term is not in index
    tweets = list(tweets)

    ranked_tweets, scores = rank_documents(query, tweets, index, idf, tf)

    return ranked_tweets, scores


def pre_process(text):
    """
    Pre-process the tweets text by
        ● Removing stop-words
        ● Tokenization
        ● Removing punctuation
        ● Stemming

    Argument:
    text -- string (text of the tweet) to be preprocessed

    Returns:
    text - list of tokens corresponding to the input text processed
    """
    text = text.lower()  # Transform in lowercase
    text = emoji.get_emoji_regexp().sub(r'', text)  # Remove emojis
    text = re.sub(r'https\S+', '', text)  # Remove urls
    text = re.sub(r'[^\w\s]', '', text)  # Remove all not alphanumerical or underscore
    text = re.sub('_', '', text)  # Remove underscore
    text = text.split()  # Tokenize the text to get a list of terms
    stop_words = set(stopwords.words("english"))
    text = list(set(text) - set(stop_words))  # remove stopwords
    text = [i for i in text if (len(i) > 0 and i != ' ')]  # Remove double spaces and empty texts
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]  # perform stemming

    return text


def create_index_tfidf(corpus):
    """
    Implement the inverted index and compute tf, df and idf

    Argument:
    tweets - array containing all the tweet objects and its information

    Returns:
    index - the inverted index (implemented through a Python dictionary) containing terms as keys and the corresponding
    list of document these keys appears in.
    tf - normalized term frequency for each term in each document
    df - number of documents each term appear in
    idf - inverse document frequency of each term
    """

    num_tweets = len(corpus)  # number of documents in the collection

    index = defaultdict(list)
    tf = defaultdict(list)  # term frequencies of terms in documents (documents in the same order as in the main index)
    df = defaultdict(int)  # document frequencies of terms in the corpus
    idf = defaultdict(float)

    for t_id in corpus.keys():

        original_text = corpus[t_id].text
        corpus[t_id].set_proc_text(pre_process(original_text))
        terms = corpus[t_id].processed_text

        current_page_index = {}
        current_page_frequencies = {}

        for term in terms:
            current_page_index[term] = [t_id]
            current_page_frequencies[term] = float(terms.count(term))

        # normalize term frequencies
        norm = 0
        for term in current_page_frequencies.keys():
            norm += current_page_frequencies[term] ** 2
        norm = math.sqrt(norm)

        # calculate the tf(dividing the term frequency by the above computed norm) and df weights
        for term in current_page_index.keys():
            # term frequency = count frequency/total count
            tf[term].append(np.round(current_page_frequencies[term] / norm, 4))
            # increment the document frequency of current term (number of documents containing the current term)
            df[term] += 1  # increment DF for current term

        # merge the current page index with the main index
        for term, t_id in current_page_index.items():
            index[term].append(t_id)

        # Compute IDF following the formula (3) above. HINT: use np.log
        for term in df:
            idf[term] = np.round(np.log(float(num_tweets / df[term])), 4)

    return index, tf, df, idf


def rank_documents(query_terms, tweet_ids, index, idf, tf):
    """
    Perform the ranking of the results of a search based on the tf-idf weights

    Argument:
    query_terms -- list of query terms
    tweet_ids -- list of tweet_ids, to rank, matching the query
    index -- inverted index data structure
    idf -- inverted document frequencies
    tf -- term frequencies

    Returns:
    The list of ranked tweets, and the list of their scores
    """

    doc_vectors = defaultdict(lambda: [0] * len(query_terms))
    query_vector = [0] * len(query_terms)

    # compute the norm for the query tf
    query_terms_count = collections.Counter(query_terms)  # get the frequency of each term in the query.

    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(query_terms):  # termIndex is the index of the term in the query
        if term not in index:
            continue

        # Compute tf*idf(normalize TF as done with documents)
        query_vector[termIndex] = query_terms_count[term] / query_norm * idf[term]

        # Generate doc_vectors for matching docs
        for tweet_index, (tweet_id) in enumerate(index[term]):
            tweet_id = tweet_id[0]
            if tweet_id in tweet_ids:
                doc_vectors[tweet_id][termIndex] = tf[term][tweet_index] * idf[term]

    # Calculate the score of each doc
    # compute the cosine similarity between queryVector and each docVector:

    tweet_scores = [[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items()]
    tweet_scores.sort(reverse=True)
    result_tweets = [x[1] for x in tweet_scores]

    tweet_scores_dict = {item[1]: item[0] for item in tweet_scores}

    result_tweets_2 = []
    tweet_scores_2 = []
    for result_id in result_tweets:
        all_words = True
        for term in query_terms:
            if [result_id] not in index[term]:
                all_words = False
        if all_words:
            result_tweets_2.append(result_id)
            tweet_scores_2.append([tweet_scores_dict[result_id], result_id])

    if len(result_tweets) == 0:
        print("No results found, try again")

    return result_tweets_2, tweet_scores_2
