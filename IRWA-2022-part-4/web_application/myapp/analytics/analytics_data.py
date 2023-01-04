import datetime
import json
import pickle
import random
from pathlib import Path

import httpagentparser
# pip install requests
import requests


class AnalyticsData:
    """
    An in memory persistence object.
    Declare more variables to hold analytics tables.
    """

    # statistics table 1
    # fact_clicks is a dictionary with the click counters: key = doc id | value = click counter
    def __init__(self, dump_path):
        self.dump_path = dump_path + '_analytics.pk'

        # storage
        # dict to store results with key = doc id and value = result_data
        self.fact_results = dict()

        # dict with key = search id and value = terms
        self.fact_queries = dict()

        # dict to count how many time a term is searched
        self.fact_terms = dict()

        # dict with key = user_context and value = (datetime, search query)
        self.fact_users = dict()

        self.ip2user = dict()
        if Path(dump_path).exists():
            print('load from previous statistics')
            self.fact_results, self.fact_queries, self.fact_terms, self.fact_users, self.ip2user = pickle.load(Path(dump_path).open('rb'))

    def save_query_terms(self, terms: str) -> int:
        for id, value in self.fact_queries.items():
            if terms == value:
                return id

        id = random.randint(0, 1000000)
        while id in self.fact_queries:
            id = random.randint(0, 1000000)
        self.fact_queries[id] = terms

        for term in terms.split():
            if term in self.fact_terms.keys():
                self.fact_terms[term]+=1
            else:
                self.fact_terms[term] = 1
        return id

    def save(self):
        print('Saving stats')
        pickle.dump((self.fact_results, self.fact_queries, self.fact_terms, self.fact_users, self.ip2user), Path(self.dump_path).open('wb'))


class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)


class user_context:
    def __init__(self, request):
        self.user_ip = request.remote_addr
        self.user_agent = request.headers.get('User-Agent')
        self.agent = httpagentparser.detect(self.user_agent)
        self.location = self.get_location()

    def get_ip(self):
        response = requests.get('https://api64.ipify.org?format=json').json()
        return response["ip"]

    def get_location(self):
        ip_address = self.get_ip()
        response = requests.get(f'https://ipapi.co/{ip_address}/json/').json()
        location_data = {
            "ip": ip_address,
            "city": response.get("city"),
            "region": response.get("region"),
            "country": response.get("country_name"),
            "continent_code": response.get("continent_code")
        }
        return location_data

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)


class results_data:
    def __init__(self):
        self.dwell_time = []
        self.clicks = dict([])
        self.query = []

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)