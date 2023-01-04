import json
import numpy as np


class Tweet:
    """
    Original corpus data as an object
    """

    def __init__(self, id, title, text, date, likes, retweets, followers, url, tw_url, hashtags, user):
        self.id = id
        self.title = title
        self.text = text
        self.date = date
        self.likes = likes
        self.retweets = retweets
        self.followers = followers
        self.url = url
        self.tw_url = tw_url
        self.hashtags = hashtags
        self.user = user

        self.processed_text = np.nan   # save space for processed text after stemming & term building

    def set_proc_text(self, p_text):
        self.processed_text = p_text

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)


class StatsTweet:
    """
    Original corpus data as an object
    """

    def __init__(self, id, title, text, date, url, count):
        self.id = id
        self.title = title
        self.text = text
        self.date = date
        self.url = url
        self.count = count

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)


class ResultItem:
    def __init__(self, id, title, text, date, url, tw_url, ranking):
        self.id = id
        self.title = title
        self.text = text
        self.date = date
        self.url = url
        self.tw_url = tw_url
        self.ranking = ranking  # score of ranking
