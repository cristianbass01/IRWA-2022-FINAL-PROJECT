import numpy as np

from myapp.search.objects import ResultItem
from myapp.search.algorithms import create_index_tfidf, search_in_corpus


class SearchEngine:
    def __init__(self, corpus):
        index, tf, df, idf = create_index_tfidf(corpus)  # create the inverted index just once and store it
        self.index = index
        self.tf = tf
        self.df = df
        self.idf = idf
        self.corpus = corpus

    def search(self, search_query, search_id):
        print("Search query:", search_query)

        results = []

        ranked_ids, score_per_id = search_in_corpus(search_query, self.index, self.idf, self.tf)

        for n in range(0, len(ranked_ids)):
            t_id = ranked_ids[n]
            tweet = self.corpus[t_id]
            rank_score = np.round(score_per_id[n][0], 2)
            results.append(ResultItem(t_id, tweet.title, tweet.text, tweet.date, tweet.url, tweet.tw_url, rank_score))

        return results
