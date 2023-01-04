import pandas as pd

from myapp.core.utils import load_json_file
from myapp.search.objects import Tweet

_corpus = {}


def load_corpus(path) -> [Tweet]:
    """
    Load file and transform to dictionary with each document as an object for easier treatment when needed for displaying
     in results, stats, etc.
    :param path:
    :return:
    """
    df = _load_corpus_as_dataframe(path)
    df.apply(_row_to_doc_dict, axis=1)
    return _corpus


def _load_corpus_as_dataframe(path):
    """
    Load documents corpus from file in 'path'
    :return:
    """
    json_data = load_json_file(path)
    tweets_df = _load_tweets_as_dataframe(json_data)
    _clean_hashtags_and_urls(tweets_df)
    # Rename columns to obtain: Tweet | Username | Date | Hashtags | Likes | Retweets | Followers | Url | Language
    corpus = tweets_df.rename(
        columns={"id": "Id", "full_text": "Tweet", "screen_name": "Username", "created_at": "Date",
                 "favorite_count": "Likes", "retweet_count": "Retweets", "lang": "Language",
                 "followers_count": "Followers"})

    # select only interesting columns
    filter_columns = ["Id", "Tweet", "Username", "Date", "Hashtags", "Likes", "Retweets", "Followers", "Url",
                      "Twitter_Url", "Language"]
    corpus = corpus[filter_columns]
    return corpus


def _load_tweets_as_dataframe(json_data):
    data = pd.DataFrame(json_data).transpose()
    # parse entities as new columns
    data = pd.concat([data.drop(['entities'], axis=1), data['entities'].apply(pd.Series)], axis=1)
    # parse user data as new columns and rename some columns to prevent duplicate column names
    data = pd.concat([data.drop(['user'], axis=1), data['user'].apply(pd.Series).rename(
        columns={"created_at": "user_created_at", "id": "user_id", "id_str": "user_id_str", "lang": "user_lang"})],
                     axis=1)
    return data


def _build_tags(row):
    tags = []
    # for ht in row["hashtags"]:
    for ht in row:
        tags.append(ht["text"])
    return tags


def _build_url(row):
    url = ""
    try:
        url = row["entities"]["url"]["urls"][0]["url"]  # tweet URL
    except:
        try:
            url = row["retweeted_status"]["extended_tweet"]["entities"]["media"][0]["url"]  # Retweeted
        except:
            url = ""
    return url


#  function to create the link to the tweet of Twitter's page
def _url_to_twitter(row):
    return 'https://twitter.com/WHO/status/' + str(row["id"])


def _clean_hashtags_and_urls(df):
    df["Hashtags"] = df["hashtags"].apply(_build_tags)
    df["Url"] = df.apply(lambda row: _build_url(row), axis=1)
    df["Twitter_Url"] = df.apply(lambda row: _url_to_twitter(row), axis=1)  # link to the titter page's post
    df.drop(columns=["entities"], axis=1, inplace=True)


def _row_to_doc_dict(row: pd.Series):
    _corpus[row['Id']] = Tweet(row['Id'], row['Tweet'][0:100], row['Tweet'], row['Date'][:-11], row['Likes'],
                               row['Retweets'], row['Followers'], row['Url'], row['Twitter_Url'], row['Hashtags'],
                               row['Username'])
