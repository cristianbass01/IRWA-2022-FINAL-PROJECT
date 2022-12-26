import datetime
import os
from json import JSONEncoder
import time

# pip install httpagentparser
import httpagentparser  # for getting the user agent as json
import nltk
from flask import Flask, render_template, session, jsonify
from flask import request

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc, user_context, results_data
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine


# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default

# end lines ***for using method to_json in objects ***

# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = 'afgsreg86sr897b6st8b76va8er76fcs6g8d7'
# open browser dev tool to see the cookies
app.session_cookie_name = 'IRWA_SEARCH_ENGINE'

# print("current dir", os.getcwd() + "\n")
# print("__file__", __file__ + "\n")
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
# print(path + ' --> ' + filename + "\n")
# load documents corpus into memory.
file_path = path + "/data/tweets-data-who.json"
dump_path = path + "/dump/tweets-data-who"
# file_path = "../../tweets-data-who.json"
corpus = load_corpus(file_path, dump_path)
print("loaded corpus. first elem:", list(corpus.values())[0])

# instantiate our search engine
search_engine = SearchEngine(corpus, dump_path)

# instantiate our in memory persistence
analytics_data = AnalyticsData(dump_path)


# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")

    if 'details' in session and 'doc_details_time' in session and session['details']:
        update_dwell()

    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests
    user_cont = user_context(request)
    user = user_cont.user_ip
    analytics_data.ip2user[user] = user_cont

    print("Remote IP: {} - JSON user browser {}".format(user, user_cont.agent))

    if user in analytics_data.fact_users.keys():
        analytics_data.fact_users[user].append([datetime.datetime.now(), []])
    else:
        analytics_data.fact_users[user] = [[datetime.datetime.now(), []]]

    return render_template('index.html', page_title="Welcome")


@app.route('/search', methods=['POST'])
def search_form_post():
    analytics_data.save()
    if 'details' in session and 'doc_details_time' in session and session['details']:
        update_dwell()

    search_query = request.form['search-query']

    session['last_search_query'] = search_query

    search_id = analytics_data.save_query_terms(search_query)

    results = search_engine.search(search_query, search_id, corpus)

    found_count = len(results)

    session['last_found_count'] = found_count
    session['last_search_query_id'] = search_id

    # time based sessions: queries of a user in the same sit down (physical
    # session).
    user_cont = user_context(request)
    user = user_cont.user_ip
    analytics_data.ip2user[user] = user_cont

    if user in analytics_data.fact_users.keys():
        analytics_data.fact_users[user][len(analytics_data.fact_users[user])-1][1].append(search_id)
    else:
        analytics_data.fact_users[user] = [[datetime.datetime.now(), [search_id]]]

    # clicks on documents.
    # to what query where related.
    # ranking of clicked documents.
    for results_doc in results:
        doc_id = results_doc.id
        ranking = results_doc.ranking
        if doc_id not in analytics_data.fact_results.keys():
            analytics_data.fact_results[doc_id] = results_data()
        analytics_data.fact_results[doc_id].query.append((search_id, ranking))

    return render_template('results.html', results_list=results, page_title="Results", found_counter=found_count)


@app.route('/doc_details', methods=['GET'])
def doc_details():
    analytics_data.save()
    # getting request parameters:
    # user = request.args.get('user')
    if 'details' in session and 'doc_details_time' in session and session['details']:
        update_dwell()

    session['details'] = True

    # get the query string parameters from request
    clicked_doc_id = int(request.args["id"])
    search_id = int(request.args["search_id"])  # transform to Integer
    p2 = int(request.args["param2"])  # transform to Integer
    print("click in id={}".format(clicked_doc_id))

    session['doc_details'] = clicked_doc_id
    session['doc_details_time'] = time.time()
    # object of clicked doc (to sent to doc_details page)
    doc = corpus[clicked_doc_id]

    # store data in statistics table 1
    if clicked_doc_id in analytics_data.fact_results.keys():
        if search_id in analytics_data.fact_results[clicked_doc_id].clicks.keys():
            analytics_data.fact_results[clicked_doc_id].clicks[search_id] += 0.5
        else:
            analytics_data.fact_results[clicked_doc_id].clicks[search_id] = 0.5
    else:
        analytics_data.fact_results[clicked_doc_id] = results_data()
        analytics_data.fact_results[clicked_doc_id].clicks[search_id] = 0.5

    print("fact_results clicks count for id={} is {}".format(clicked_doc_id,
                                                             analytics_data.fact_results[clicked_doc_id].clicks))

    return render_template('doc_details.html', doc=doc, page_title="Details")


def update_dwell():
    doc = session['doc_details']
    last_time = session['doc_details_time']
    current_time = time.time()
    if doc not in analytics_data.fact_results.keys():
        analytics_data.fact_results[doc] = results_data()
    analytics_data.fact_results[doc].dwell_time.append(current_time - last_time)
    session['details'] = False


@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example. ### Replace with dashboard ###
    :return:
    """
    analytics_data.save()
    if 'details' in session and 'doc_details_time' in session and session['details']:
        update_dwell()

    docs = []
    # ### Start replace with your code ###

    for doc_id in analytics_data.fact_results.keys():
        row: Document = corpus[int(doc_id)]
        count = sum(list(analytics_data.fact_results[doc_id].clicks.values()))
        if count >0:
            doc = StatsDocument(row.id, row.title, row.description, row.doc_date, row.url, count)
            docs.append(doc)
        # simulate sort by ranking
    docs.sort(key=lambda doc: doc.count, reverse=True)

    users = []
    for user in analytics_data.fact_users:
        data = analytics_data.fact_users[user][len(analytics_data.fact_users[user])-1][0]
        queries = [analytics_data.fact_queries[query] for query in analytics_data.fact_users[user][len(analytics_data.fact_users[user])-1][1]]
        users.append((user, data, queries))

    terms = sorted(analytics_data.fact_terms.items(), key=lambda x: x[1], reverse=True)

    return render_template('stats.html', clicks_data=docs[:10], users=users[:10], terms= terms[:10], page_title='Statistics')
    # ### End replace with your code ###


@app.route('/dashboard', methods=['GET'])
def dashboard():
    analytics_data.save()
    if 'details' in session and 'doc_details_time' in session and session['details']:
        update_dwell()

    visited_docs = []
    print(analytics_data.fact_results.keys())
    for doc_id in analytics_data.fact_results.keys():
        d: Document = corpus[int(doc_id)]
        doc = ClickedDoc(doc_id, d.description, sum(list(analytics_data.fact_results[doc_id].clicks.values())))
        if doc.counter > 0:
            visited_docs.append(doc)

    # simulate sort by ranking
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)
    visited_ser = []
    for doc in visited_docs:
        visited_ser.append(doc.to_json())

    terms = sorted(analytics_data.fact_terms.items(),  key=lambda x: x[1], reverse=True)
    terms_ser = []
    for term in terms:
        terms_ser.append((term[0], term[1]))

    return render_template('dashboard.html', visited_docs=visited_ser[:10], visited_terms= terms_ser, page_title='Dashboard')


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=True)
