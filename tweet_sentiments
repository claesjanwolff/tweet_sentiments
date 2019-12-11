import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer, PatternAnalyzer
from tweepy import OAuthHandler
from ftfy import fix_text
import json
import os
import string
import matplotlib.pyplot as plt
import pandas as pd
import regex as re
import seaborn as sns
import tweepy
import tqdm

consumer_key = "your twitter dev consumer key here"
consumer_secret = "your twitter dev consumer secret here"
access_key = "your twitter dev access key here"
access_secret = "your twitter dev access secret here"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

filepath = "C:\\Downloads\\"
delimiter = ';'
start_date = '2019-01-01'
end_date = '2019-12-31'
batch_count = 200
max_tweets = 500

positive_emoticons = {':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}', ':^)', ':-D', ':D',
                      '8-D',
                      '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D', '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*',
                      '>:P',
                      ':-P', ':P', 'X-P', 'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)', '<3'}

negative_emoticons = {':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<', ':-[', ':-<', '=\\', '=/',
                      '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c', ':c', ':{', '>:\\', ';('}

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

emoticons = positive_emoticons.union(negative_emoticons)


def lemmatize(tweet_list):
    lem = WordNetLemmatizer()
    normalized_tweet = []

    for word in tweet_list:
        normalized_text = lem.lemmatize(word, 'v')
        normalized_tweet.append(normalized_text)

    return normalized_tweet


def clean_tweets(tweet, mode='string', normalize=True):
    words = fix_text(tweet)
    word_list = words.split()

    tweet_list = [word.lower() for word in word_list]
    tweet_list = [word for word in tweet_list if word != 'user']
    tweet_list = [word for word in tweet_list if re.match(r'[^\W\d]*$', word)]
    tweet_list = [word for word in tweet_list if re.sub(r':', '', word)]
    tweet_list = [word for word in tweet_list if re.sub(r'rt', '', word)]
    tweet_list = [word for word in tweet_list if re.sub(r'fav', '', word)]
    tweet_list = [word for word in tweet_list if re.sub(r'‚Ä¶', '', word)]
    tweet_list = [word for word in tweet_list if emoji_pattern.sub(r'', word)]
    tweet_list = [word for word in tweet_list if word not in emoticons]

    if normalize:
        tweet_list = [word for word in tweet_list if word not in string.punctuation]
        tweet_list = [word for word in tweet_list if word not in stopwords.words('english')]
        tweet_list = lemmatize(tweet_list)

    if mode == 'list':
        return tweet_list
    else:
        return ' '.join(tweet_list)


def get_sentiment(text, analyzer=NaiveBayesAnalyzer()):
    # text_blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
    text_blob = TextBlob(text, analyzer=analyzer)
    text_blob = text_blob.correct()
    sentiment = text_blob.sentiment
    polarity = sentiment.polarity
    subjectivity = sentiment.subjectivity

    if polarity > 0:
        label = 'positive'
    elif polarity == 0:
        label = 'neutral'
    else:
        label = 'negative'

    return label, polarity, subjectivity


def get_all_tweets_from(screen_name, startdate=None, enddate=None, exclude_replies=True, retweeted=True, language='en'):
    cols = ['original_author', 'id', 'created_at', 'source', 'original_text', 'clean_text', 'word_list', 'clean_words',
            'language', 'favorite_count', 'retweet_count', 'possibly_sensitive', 'hashtags',
            'user_mentions', 'location_boundaries', 'location', 'polarity', 'subjectivity', 'label']

    all_tweets = []

    tweets_count = api.get_user(screen_name).statuses_count

    for tweet in tqdm.tqdm(
            iterable=tweepy.Cursor(api.user_timeline,
                                   screen_name=screen_name,
                                   tweet_mode='extended',
                                   count=batch_count,
                                   #retweeted=retweeted,
                                   lang=language,
                                   #include_rts=False,
                                   #exclude_replies=exclude_replies,
                                   include_entities=True,
                                   since=startdate).items(tweets_count), total=tweets_count):
        single_tweet = tweet._json

        clean_text = clean_tweets(tweet=single_tweet['full_text'], mode='string', normalize=True)
        clean_text_list = clean_tweets(tweet=single_tweet['full_text'], mode='list', normalize=True)
        clean_words = clean_tweets(tweet=single_tweet['full_text'], mode='string', normalize=False)

        try:
            sensitivity = single_tweet['possibly_sensitive']
        except:
            sensitivity = None

        try:
            hashtags = ", ".join([hashtag_item['text'] for hashtag_item in single_tweet['entities']['hashtags']])
            user_mentions = ", ".join([mention['screen_name'] for mention in single_tweet['entities']['user_mentions']])
        except:
            continue

        try:
            user_coordinates = [coord for loc in single_tweet['place']['bounding_box']['coordinates'] for coord in loc]
        except:
            user_coordinates = None

        try:
            user_location = single_tweet['user']['location']
        except:
            user_location = ''

        label, polarity, subjectivity = get_sentiment(clean_words)

        all_tweets.append([single_tweet['user']['screen_name'],
                           single_tweet['id'],
                           single_tweet['created_at'],
                           single_tweet['source'],
                           single_tweet['full_text'],
                           clean_text,
                           clean_text_list,
                           clean_words,
                           single_tweet['lang'],
                           single_tweet['favorite_count'],
                           single_tweet['retweet_count'],
                           sensitivity,
                           hashtags,
                           user_mentions,
                           user_coordinates,
                           user_location,
                           polarity,
                           subjectivity,
                           label])

    dftweets = pd.DataFrame(data=all_tweets, columns=cols)
    print("write " + screen_name + '_tweets.csv' + " to " + filepath)
    dftweets.to_csv(filepath + screen_name + '_tweets.csv', encoding='utf-8', index=False, sep=delimiter)

    return dftweets


def get_all_tweets_with(keyword, startdate=None, enddate=None, exclude_replies=True, retweeted=True, language='en'):
    cols = ['original_author', 'id', 'created_at', 'source', 'original_text', 'clean_text', 'word_list', 'clean_words',
            'language', 'favorite_count', 'retweet_count', 'possibly_sensitive', 'hashtags',
            'user_mentions', 'location_boundaries', 'location', 'polarity', 'subjectivity', 'label']

    all_tweets = []
    cnt = 0

    for tweet in tqdm.tqdm(iterable=tweepy.Cursor(api.search, tweet_mode='extended', q=keyword, count=batch_count,
                                                  retweeted=retweeted,
                                                  lang=language,
                                                  include_rts=False,
                                                  exclude_replies=exclude_replies,
                                                  include_entities=True,
                                                  since=startdate).items(max_tweets), total=max_tweets):
        single_tweet = tweet._json

        clean_text = clean_tweets(tweet=single_tweet['full_text'], mode='string', normalize=True)
        clean_text_list = clean_tweets(tweet=single_tweet['full_text'], mode='list', normalize=True)
        clean_words = clean_tweets(tweet=single_tweet['full_text'], mode='string', normalize=False)

        try:
            sensitivity = single_tweet['possibly_sensitive']
        except:
            sensitivity = None

        try:
            hashtags = ", ".join([hashtag_item['text'] for hashtag_item in single_tweet['entities']['hashtags']])
            user_mentions = ", ".join([mention['screen_name'] for mention in single_tweet['entities']['user_mentions']])
        except:
            continue

        try:
            user_coordinates = [coord for loc in single_tweet['place']['bounding_box']['coordinates'] for coord in loc]
        except:
            user_coordinates = None

        try:
            user_location = single_tweet['user']['location']
        except:
            user_location = ''

        label, polarity, subjectivity = get_sentiment(clean_words)

        all_tweets.append([single_tweet['user']['screen_name'],
                           single_tweet['id'],
                           single_tweet['created_at'],
                           single_tweet['source'],
                           single_tweet['full_text'],
                           clean_text,
                           clean_text_list,
                           clean_words,
                           single_tweet['lang'],
                           single_tweet['favorite_count'],
                           single_tweet['retweet_count'],
                           sensitivity,
                           hashtags,
                           user_mentions,
                           user_coordinates,
                           user_location,
                           polarity,
                           subjectivity,
                           label])

        if len(all_tweets) % 100 == 0:
            print('...%s tweets have been downloaded so far' % len(all_tweets))

    dftweets = pd.DataFrame(data=all_tweets, columns=cols)
    keyword_no_hashtag = keyword.replace('#', '')
    print("write " + keyword_no_hashtag + '_tweets.csv' + " to " + filepath)
    dftweets.to_csv(filepath + keyword_no_hashtag + '_tweets.csv', encoding='utf-8', index=False, sep=delimiter)

    return dftweets


def plot_tweets(df):
    result_cnt = (df['label']).count()
    pos_cnt = (df['label'] == 'positive').sum()
    neu_cnt = (df['label'] == 'neutral').sum()
    neg_cnt = (df['label'] == 'negative').sum()

    print("Positive tweets percentage: {} %".format(100 * pos_cnt / result_cnt))
    print("Neutral tweets percentage: {} %".format(100 * neu_cnt / result_cnt))
    print("Negative tweets percentage: {} %".format(100 * neg_cnt / result_cnt))

    sns.countplot(x='label', data=df)
    plt.show()

    sns.boxplot(x='location', y='retweet_count', hue='label', data=df)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    df['polarity'].hist(bins=[-1, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1], ax=ax, color="purple")
    plt.title("Sentiments Polarity")
    plt.show()


if __name__ == '__main__':
    df_user_tweets = get_all_tweets_from('apocalypticafi')
    plot_tweets(df_user_tweets)

    # df_all_tweets = get_all_tweets_with('#python')
    # plot_tweets(df_all_tweets)
