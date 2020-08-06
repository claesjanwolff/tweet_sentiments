import nltk


from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer, PatternAnalyzer
from wordcloud import WordCloud
from tweepy import OAuthHandler
from ftfy import fix_text
from tqdm import trange
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.model_selection import GridSearchCV, BaseCrossValidator, RandomizedSearchCV
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, make_scorer, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Embedding, GlobalMaxPool1D, Flatten, GlobalMaxPooling1D
from keras.layers import Dropout, BatchNormalization, Input, LSTM, GRU, Activation
from keras.layers import Conv1D, MaxPooling1D, Bidirectional, LeakyReLU, SimpleRNN, SpatialDropout1D
from keras.optimizers import Adam, RMSprop, TFOptimizer, SGD
from keras.callbacks import History, EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.initializers import Constant
from subprocess import call
from pydotplus import graph_from_dot_data, graph_from_dot_file

import pydotplus
import six
import keras
import json
import os
import string
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regex as re
import seaborn as sns
import tweepy
import tqdm
import pydot
import treeinterpreter

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

print(keras.backend.backend())

consumer_key = "<your consumer key here>"
consumer_secret = "<your consumer secret here>"
access_key = "<your access key here>"
access_secret = "<your access secret>"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

filepath = "C:\\Downloads\\"
delimiter = ';'
start_date = '2019-01-01'
end_date = '2019-12-31'
batch_count = 200
max_tweets = 5000

cols = ['original_author', 'id', 'created_at', 'source', 'original_text', 'clean_text', 'word_list', 'clean_words',
        'language', 'favorite_count', 'retweet_count', 'possibly_sensitive', 'hashtags',
        'user_mentions', 'location_boundaries', 'location', 'polarity', 'subjectivity', 'label']

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
print(emoji_pattern)
emoticons = positive_emoticons.union(negative_emoticons)


def emoji_score(tweet):
    colnames = ["Char", "Image", "Unicode", "Occurrences", "Position", "Neg", "Neut", "Pos", "Sentiment score"]
    dfemoji = pd.read_csv(filepath + "\\emoji_score.csv", header=0, delimiter=delimiter, index_col=0, names=colnames)
    emoji_list = dfemoji['Unicode'].values
    sentiment_list = []

    cnt = 0
    for e in tqdm.tqdm(tweet, desc="emoji " + str(cnt), total=len(tweet)):
        emoji_decode = f'0x{ord(e):X}'.lower()
        if emoji_decode in emoji_list:
            cnt += 1
            dfsentiment = dfemoji.loc[dfemoji['Unicode'] == emoji_decode, 'Sentiment score']
            sentiment_list.append((dfsentiment.values / 10000))

    if len(sentiment_list) > 0:
        return np.mean(sentiment_list)
    else:
        return 0


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


def get_hashtags(dfcol):
    newtaglist = []
    hashdict = {}
    hashlist = dfcol.values.tolist()
    hashlist = [x for x in hashlist if str(x) != 'nan']
    for i, tags in enumerate(hashlist):
        for tag in tags.split():
            newtaglist.append(tag.replace(',',''))

    newtaglist.sort()

    for word in newtaglist:
        hashdict.setdefault(word, 0)
        hashdict[word] += 1

    dfhashtag = pd.DataFrame.from_dict(data=hashdict, orient='index').reset_index()
    dfhashtag.columns = ['hashtag', 'count']

    return hashlist, hashdict, dfhashtag


def flatten(nested_list, new_list):
    for i in nested_list:
        if isinstance(i, list):
            flatten(i, new_list)
        else:
            if str(i) != 'nan':
                new_list.append(str(i).replace(',', ''))

    return new_list


def get_freq(text_list):
    flat_list = flatten(text_list, [])
    word_list = []

    for i, sent in enumerate(flat_list):
        try:
            word = sent.split()
            word_list.extend(word)
        except AttributeError:
            print("error for ", word)
            word = str(word).split()
            word_list.extend(word)
            continue

    word_freq = pd.Series(word_list).value_counts()
    return word_freq


def get_sentiment(dirty_tweet, clean_tweet, analyzer=NaiveBayesAnalyzer()):
    text_blob = TextBlob(clean_tweet, analyzer=analyzer)
    text_blob = text_blob.correct()
    sentiment = text_blob.sentiment
    polarity = sentiment.polarity
    subjectivity = sentiment.subjectivity

    added_weight = emoji_score(dirty_tweet)

    # if added_weight != 0:
    #    print("adding emoji sentiment ", added_weight)

    polarity = polarity + added_weight

    if polarity > 0:
        label = 'positive'
    elif polarity == 0:
        label = 'neutral'
    else:
        label = 'negative'

    return label, polarity, subjectivity


def get_all_tweets(key, value, startdate=None, enddate=None, exclude_replies=False, retweeted=True, language='en'):
    print("using: " + key + " = " + value)

    if key.lower() == 'user':
        tweet_count = api.get_user(value).statuses_count

        cursor = tweepy.Cursor(api.user_timeline,
                               screen_name=screen_name,
                               tweet_mode='extended',
                               count=batch_count,
                               # retweeted=retweeted,
                               lang=language,
                               # include_rts=False,
                               exclude_replies=exclude_replies,
                               include_entities=True,
                               since=startdate).items(tweet_count)
    elif key.lower() == 'query':
        tweet_count = max_tweets

        cursor = tweepy.Cursor(api.search,
                               q=value,
                               count=batch_count,
                               retweeted=retweeted,
                               lang=language,
                               include_rts=False,
                               exclude_replies=exclude_replies,
                               include_entities=True,
                               tweet_mode='extended',
                               since=startdate).items(tweet_count)

    all_tweets = []

    for tweet in tqdm.tqdm(iterable=cursor, total=tweet_count, desc="tweet"):
        single_tweet = tweet._json

        clean_text = clean_tweets(tweet=single_tweet['full_text'], mode='string', normalize=True)
        clean_text_list = clean_tweets(tweet=single_tweet['full_text'], mode='list', normalize=True)
        clean_words = clean_tweets(tweet=single_tweet['full_text'], mode='string', normalize=False)

        try:
            sensitivity = single_tweet['possibly_sensitive']
        except Exception as esens:
            sensitivity = None
            continue

        try:
            hashtags = ", ".join([hashtag_item['text'] for hashtag_item in single_tweet['entities']['hashtags']])
            hashtags = hashtags.lower()
            user_mentions = ", ".join([mention['screen_name'] for mention in single_tweet['entities']['user_mentions']])
        except Exception as ehash:
            continue

        try:
            user_coordinates = [coord for loc in single_tweet['place']['bounding_box']['coordinates'] for coord in loc]
        except Exception as ecoord:
            user_coordinates = None
            continue

        try:
            user_location = single_tweet['user']['location']
        except Exception as eloc:
            user_location = ''
            continue

        # choose between NaiveBayesAnalyzer and PatternAnalyzer
        label, polarity, subjectivity = get_sentiment(dirty_tweet=single_tweet['full_text'], clean_tweet=clean_words,
                                                      analyzer=NaiveBayesAnalyzer())

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
    dftweets = dftweets[dftweets['clean_text'].astype(bool)]

    keyword = (key + "_" + value.replace('#', '')).lower()
    print("write " + keyword + '_tweets.csv' + " to " + filepath)
    try:
        dftweets.to_csv(filepath + keyword + '_tweets.csv', encoding='utf-8', index=False, sep=delimiter)
    except PermissionError:
        print("can't save file to disk, close file " + filepath + keyword + '_tweets.csv' + " first!")

    return dftweets


def load_tweets(key, value):
    file = (key + '_' + value.replace('#', '')).lower() + '_tweets.csv'
    print("read " + filepath + file)

    df = pd.read_csv(filepath + file, names=cols, index_col=False, delimiter=delimiter, header=0)
    return df


def plot_val_curve(model, x, y):
    param_range = np.arange(1, 2500, 2)

    # Calculate accuracy on training and test set using range of parameter values
    train_scores, test_scores = validation_curve(model,
                                                 x,
                                                 y,
                                                 param_name="n_estimators",
                                                 param_range=param_range,
                                                 cv=5,
                                                 scoring="accuracy",
                                                 n_jobs=-1)

    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, label="Training score", color="black")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

    # Plot accurancy bands for training and test sets
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

    # Create plot
    plt.title("Validation Curve With Random Forest")
    plt.xlabel("Number Of Trees")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()


def plot_feature_importance(classifier, cols):
    y = classifier.feature_importances_

    fig, ax = plt.subplots()
    width = 0.4  # the width of the bars
    ind = np.arange(len(y))  # the x locations for the groups
    ax.barh(ind, y, width, color='green')
    ax.set_yticks(ind + width / 10)
    ax.set_yticklabels(cols, minor=False)
    plt.title('Feature importance in RandomForest Classifier')
    plt.xlabel('Relative importance')
    plt.ylabel('feature')
    plt.figure(figsize=(5, 5))
    fig.set_size_inches(6.5, 4.5, forward=True)
    plt.show()


def plot_feature_contribution(classifier, xtest):
    # prediction, bias, contributions = ti.predict(classifier, xtest[6:7])
    prediction, bias, contributions = ti.predict(classifier, xtest)
    N = classes + 1  # no of entries in plot , 4 ---> features & 1 ---- class label

    negative = []
    neutral = []
    positive = []

    for j in range(classes):
        list_ = [negative, neutral, positive]
        for i in range(4):
            val = contributions[0, i, j]
            list_[j].append(val)

        negative.append(prediction[0, 0] / 5)
        neutral.append(prediction[0, 1] / 5)
        positive.append(prediction[0, 2] / 5)

    fig, ax = plt.subplots()
    ind = np.arange(N)
    width = 0.15
    p1 = ax.bar(ind, negative, width, color='red', bottom=0)
    p2 = ax.bar(ind + width, neutral, width, color='green', bottom=0)
    p3 = ax.bar(ind + (2 * width), positive, width, color='yellow', bottom=0)
    ax.set_title('Contribution of all feature for a particular sentiment ')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(col, rotation=90)
    ax.legend((p1[0], p2[0], p3[0]), ('negative', 'neutral', 'positive'), bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.autoscale_view()
    plt.show()


def plot_classifier(model, x, y, ax=None, cmap='jet'):
    try:
        ax = ax or plt.gca()

        # Plot the training points
        # print("X 0", x[:, 0])
        # print("X 1", x[:, 1])

        ax.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap=cmap,
                   clim=(y.min(), y.max()), zorder=3)
        ax.axis('tight')
        ax.axis('off')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        model.fit(x, y)
        xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                             np.linspace(*ylim, num=200))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        # Create a color plot with the results
        n_classes = len(np.unique(y))
        contours = ax.contourf(xx, yy, Z, alpha=0.3,
                               levels=np.arange(n_classes + 1) - 0.5,
                               cmap=cmap, clim=(y.min(), y.max()),
                               zorder=1)

        ax.set(xlim=xlim, ylim=ylim)
        plt.show()
    except Exception as e:
        print("plot_classifier error ", str(e))


def plot_hashtag(df, searchval=None, omit=True, number=11):
    hashlist, hashdict, hashdf = get_hashtags(df["hashtags"])

    if omit:
        df1 = hashdf.sort_values('count', ascending=False).groupby('hashtag').head(number-1)
        df1 = df1[df1["hashtag"] != searchval]
    else:
        df1 = hashdf.sort_values('count', ascending=False).groupby('hashtag').head(number)

    df1 = df1.head(number)
    print(df1)

    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    sns.set(context='notebook', style='darkgrid', palette='muted', font='sans-serif', font_scale=0.8, color_codes=True,
                rc=None)
    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    sns.scatterplot(data=df1, x='hashtag', y='count', palette=cmap, size='count', sizes=(20, 200))
    if omit:
        plt.title('hashtag count minus keysearch ' + searchval)
    else:
        plt.title('hashtag count for keysearch ' + searchval)

    plt.show()


def plot_tweets(df):
    result_cnt = (df['label']).count()
    pos_cnt = (df['label'] == 'positive').sum()
    neu_cnt = (df['label'] == 'neutral').sum()
    neg_cnt = (df['label'] == 'negative').sum()

    print("Positive tweets percentage: {} %".format(100 * pos_cnt / result_cnt))
    print("Neutral tweets percentage: {} %".format(100 * neu_cnt / result_cnt))
    print("Negative tweets percentage: {} %".format(100 * neg_cnt / result_cnt))

    word_frequency = get_freq(df["clean_text"].values.tolist())
    wc = WordCloud(width=400, height=330, max_words=100, background_color='white').generate_from_frequencies(
        word_frequency)
    plt.figure(figsize=(12, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.title('most frequent words in tweets top 100', loc='left')
    plt.axis('off')
    plt.show()

    hashtag_frequency = get_freq(df["hashtags"].values.tolist())
    wc = WordCloud(width=400, height=330, max_words=100, background_color='white').generate_from_frequencies(
        hashtag_frequency)
    plt.figure(figsize=(12, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.title('most frequent hashtags in tweets top 100', loc='left')
    plt.axis('off')
    plt.show()

    sns.countplot(data=df, x="label")
    plt.title('sentiment count', loc='left')
    plt.show()

    sns.set(context='notebook', style='darkgrid', palette='muted', font='sans-serif', font_scale=0.8, color_codes=True,
            rc=None)

    sns.relplot(x="polarity", y="retweet_count", data=df, size="retweet_count", kind="scatter", sizes=(10, 100))
    plt.title('retweet count', loc='left')
    plt.show()

    try:
        sns.distplot(df["polarity"], bins=[-1, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1], norm_hist=True)
        plt.title('sentiment polarity distribution', loc='left')
        plt.show()
    except Exception as e:
        print("something went wrong, polarity must be between -1 and 1 ", str(e))
        sns.distplot(df["polarity"], bins=[-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
        plt.title('sentiment polarity distribution', loc='left')
        plt.show()


    # lm = sns.lmplot(x = "totalX", y = "NormI", hue = "Data Type", data = df, palette="Set1", legend_out=False, scatter_kws={"s": 100})


def plot_history(history, model):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.xlabel('epoch')
    plt.title('Training and validation accuracy ' + model)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.xlabel('epoch')
    plt.title('Training and validation loss ' + model)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # test emoji score:
    # print(emoji_score("Liberals ruin another Christmas classic.🙄😡"))

    search_key = 'query'
    search_val = '#climatechange'
    doc = filepath + (search_key.lower() + '_' + search_val.replace('#', '') + '_tweets.csv').lower()

    if os.path.exists(doc):
        df_all_tweets = load_tweets(key=search_key, value=search_val)
    else:
        df_all_tweets = get_all_tweets(key=search_key, value=search_val)

    plot_tweets(df_all_tweets)
    plot_hashtag(df_all_tweets, searchval=search_val.replace('#', ''), omit=True, number=11)
