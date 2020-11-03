import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from score_tweet import *
from textutil import *
from sklearn.pipeline import Pipeline, FeatureUnion


def load():
    pos_train_file = 'corpus_v4/split/tweets_pos_train.txt'
    neg_train_file = 'corpus_v4/split/tweets_neg_train.txt'

    pos_test_file = 'corpus_v4/split/tweets_pos_test.txt'
    neg_test_file = 'corpus_v4/split/tweets_neg_test.txt'

    pos_train_data, pos_train_labels = read_data(pos_train_file, 'pos')
    neg_train_data, neg_train_labels = read_data(neg_train_file, 'neg')

    pos_test_data, pos_test_labels = read_data(pos_test_file, 'pos')
    neg_test_data, neg_test_labels = read_data(neg_test_file, 'neg')
    print('------------------------------------')

    sample_size = 2
    print('{} random train tweets (positive) .... '.format(sample_size))
    print(np.array(random.sample(pos_train_data, sample_size)))
    print('------------------------------------')
    print('{} random train tweets (negative) .... '.format(sample_size))
    print(np.array(random.sample(neg_train_data, sample_size)))
    print('------------------------------------')

    x_train = pos_train_data + neg_train_data
    y_train = pos_train_labels + neg_train_labels

    x_test = pos_test_data + neg_test_data
    y_test = pos_test_labels + neg_test_labels

    print('train data size:{}\ttest data size:{}'.format(len(y_train), len(y_test)))
    print('train data: # of pos:{}\t# of neg:{}\t'.format(y_train.count('pos'), y_train.count('neg')))
    print('test data: # of pos:{}\t# of neg:{}\t'.format(y_test.count('pos'), y_test.count('neg')))
    print('------------------------------------')
    return x_train, y_train, x_test, y_test


class LexSentiScoreExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def tweet_score(self, tweet, method='base_emoji'):
        positive_score = 0
        negative_score = 0
        tweet = '- ' + tweet + ' -'
        words = tweet.split()
        # tweet_windows = list(window(words, 2))
        tweet_windows = list(window(words, 3))

        # check in special lexicon
        if contain_special_lex(tweet):
            negative_score = -2.0
            print('contain negative phrase')
        elif contain_pos_phrase(tweet):
            positive_score = 2.0
            print('contain positive phrase')
        else:
            # for prev_word, word in tweet_windows:
            tweet_info = []
            word_score = 0.0
            sent_word = ''
            for prev_word, word, next_word in tweet_windows:
                light_word = light_stem_word(word)
                ####################################
                if method == 'base':
                    word_score, sent_word = get_score_base_lex(word)
                elif method == 'emoji':
                    word_score, sent_word = get_score_emoji(word)
                elif method == 'base_emoji':
                    word_score, sent_word = get_score_base_emoji(word)
                    tweet_info.append([word, sent_word, word_score])
                elif method == 'very_lex':
                    word_score, sent_word = get_score_very_lex(word)
                    tweet_info.append([word, sent_word, word_score])
                elif method == 'very_lex_emoji':
                    word_score, sent_word = get_score_very_lex_emoji(word)
                    tweet_info.append([word, sent_word, word_score])
                elif method == 'base_lex_consider_support':
                    word_score, sent_word = get_score_base_lex_consider_support(prev_word, word, next_word)
                    tweet_info.append([word, sent_word, word_score])
                elif method == 'base_lex_consider_nag':
                    word_score, sent_word = get_score_base_lex_consider_nag(prev_word, word)
                    tweet_info.append([word, sent_word, word_score])
                elif method == 'base_light_lex':
                    word_score, sent_word = get_score_base_light_lex(word, light_word)
                    tweet_info.append([word, sent_word, word_score])
                elif method == 'base_light_lex_very':
                    word_score, sent_word = get_score_base_light_lex_very(word, light_word)
                    tweet_info.append([word, sent_word, word_score])
                elif method == 'base_light_lex_consider_nag_very':
                    word_score, sent_word = get_score_base_light_lex_consider_nag_very(prev_word, word, light_word)
                    tweet_info.append([word, sent_word, word_score])
                elif method == 'all_levels':
                    word_score, sent_word = get_score_all_levels(prev_word, word, next_word, light_word)
                    tweet_info.append([word, sent_word, word_score])

                if word_score > 0:
                    positive_score += word_score
                elif word_score < 0:
                    negative_score += word_score

        tweet_score = positive_score + negative_score
        return tweet_score

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df['score'].apply(self.tweet_score)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


def do_sa_ml(n, my_classifier, my_data):
    x_train, y_train, x_test, y_test = my_data
    print('parameters')
    print('n grams:', n)
    print('classifier:', my_classifier.__class__.__name__)
    print('------------------------------------')

    pipeline = Pipeline([
        ('vect', TfidfVectorizer(min_df=5, max_df=0.95,
                                 analyzer='word', lowercase=False,
                                 ngram_range=(1, n))),
        ('clf', my_classifier),
    ])

    pipeline.fit(x_train, y_train)
    feature_names = pipeline.named_steps['vect'].get_feature_names()
    print('features:', )

    y_predicted = pipeline.predict(x_test)
    return y_predicted
