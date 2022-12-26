import pickle
import random
from pathlib import Path

from myapp.search.algorithms import create_tfidf_index, search_in_corpus
from myapp.search.load_corpus import always_preprocess
from myapp.search.objects import ResultItem, Document


def build_results(corpus: dict, doc_scores, search_id):
    """
    Helper method, just to demo the app
    :return: a list of demo docs sorted by ranking
    """
    res = []
    for index in range(len(doc_scores)):
        doc_id = doc_scores[index][1]
        ranking = doc_scores[index][0]
        item: Document = corpus[doc_id]
        res.append(ResultItem(item.id, item.title, item.description, item.doc_date,
                              "doc_details?id={}&search_id={}&param2=2".format(item.id, search_id), ranking, item.link))

    # for index, item in enumerate(corpus['Id']):
    #     # DF columns: 'Id' 'Tweet' 'Username' 'Date' 'Hashtags' 'Likes' 'Retweets' 'Url' 'Language'
    #     res.append(DocumentInfo(item.Id, item.Tweet, item.Tweet, item.Date,
    #                             "doc_details?id={}&search_id={}&param2=2".format(item.Id, search_id), random.random()))

    # simulate sort by ranking
    return res


class SearchEngine:
    """educational search engine"""

    def __init__(self, corpus, path):
        self.corpus = corpus
        dump_file = Path(path + '_index.pk')
        if not always_preprocess and dump_file.exists():
            print("Load index data from cache!")
            self.index, self.tf, self.df, self.idf = pickle.load(dump_file.open('rb'))
        else:
            self.index, self.tf, self.df, self.idf = create_tfidf_index(corpus)
            pickle.dump([self.index, self.tf, self.df, self.idf], dump_file.open('wb'))

    def search(self, search_query, search_id, corpus):
        print("Search query:", search_query)

        doc_scores = search_in_corpus(corpus, search_query, self.index, self.idf, self.tf)
        results = build_results(self.corpus, doc_scores, search_id)

        return results