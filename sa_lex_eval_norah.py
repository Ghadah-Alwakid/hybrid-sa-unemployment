import sys

import numpy as np
from pycm import ConfusionMatrix

from score_tweet import *


def load_tweets(corpus_dir):
    # corpus_dir = './corpus_v4/split/'

    pos_tweets = load_corpus(corpus_dir + 'PosTest.txt', sep=False)
    neg_tweets = load_corpus(corpus_dir + 'NegTest.txt', sep=False)
    corpus = [(tweet, 'pos') for tweet in pos_tweets]
    corpus.extend([(tweet, 'neg') for tweet in neg_tweets])
    return corpus


def classify_tweet(tweet, method):
    positive_score = 0
    negative_score = 0
    positive_sentiments = list()
    negative_sentiments = list()
    features = find_features_in_text(tweet, features_dict)
    tweet = '- ' + tweet + ' -'
    words = tweet.split()
    # tweet_windows = list(window(words, 2))
    tweet_windows = list(window(words, 3))
    print('tweet: {}'.format(tweet.strip()))
    print('tweet contains negation: {}'.format(contains_negation(tweet, negation_list)))
    print('tweet contains support words: {}'.format(contains_support_words(tweet, support_list)))

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
                positive_sentiments.append(sent_word)
                positive_score += word_score
            elif word_score < 0:
                negative_sentiments.append(sent_word)
                negative_score += word_score

            feat = get_feature_from_word(word, features_dict)
            if feat:
                features.add((word, feat))
            else:
                feat = get_feature_from_word(light_word, features_dict)
                if feat:
                    features.add((word, feat))

        # print(tabulate(tweet_info, headers=['Score', 'Sentiment', 'Word'], tablefmt='pipe'))
        for item in tweet_info:
            if item[1] is None:
                print('word: {0: <8}\tsentiment:{1: <12}\tscore {2:3}'.format(item[0], '', item[2]))
            else:
                print('word: {0: <8}\tsentiment:{1: <12}\tscore:{2:3}'.format(item[0], item[1], item[2]))

    ###########################

    label = get_label(pos_score=positive_score, neg_score=negative_score)

    tweet_score = positive_score + negative_score
    print('score:', tweet_score)
    print('positive sentiments:', positive_sentiments)
    print('negative sentiments:', negative_sentiments)
    if features:
        print('features:', features)
    # print('------------------------\n\n')
    return label, tweet_score, positive_sentiments, negative_sentiments, features


def do_sa_lex(tweets_corpus, method):
    prediction_list = list()
    y_list = list()
    feature_list = set()
    for tweet, label in tweets_corpus:
        result = classify_tweet(tweet, method)
        pred, score, pos_sent, neg_sent, features = result
        if features:
            for f in features:
                feature_list.add(f)
        print('predicted:', pred)
        print('actual label:', label)
        print('label == predicted?', label == pred)
        print('============================\n\n')
        prediction_list.append(pred)
        y_list.append(label)

    print('features mentioned in the corpus:')
    for f in feature_list:
        print(f)

    label_names = ['neg', 'pos']
    pred_labels = np.asarray([label_names.index(p) for p in prediction_list])
    # return pred_labels
    return prediction_list
    # true_labels = np.asarray([label_names.index(y) for y in y_list])
    # cm = ConfusionMatrix(actual_vector=true_labels, predict_vector=pred_labels)  # Create CM From Data
    #
    # print('----------- summary results -----------------')
    # print('classes:\n', cm.classes)
    # print('classes names: ', *label_names)
    # print('ACC(Accuracy)', cm.class_stat.get('ACC'))
    # print('F1 score', cm.class_stat.get('F1'))
    # print('Accuracy AVG', sum(cm.class_stat.get('ACC').values()) / len(cm.class_stat.get('ACC')))
    # print('F1 AVG', sum(cm.class_stat.get('F1').values()) / len(cm.class_stat.get('F1')))
    # print('----------------------------------------------')


if __name__ == '__main__':
    methods = (
        'base',  # base lexicon
        'emoji',  # only emoji
        'base_emoji',  # base + emoji
        'very_lex',  # base + very
        'very_lex_emoji',  # base + very + emoji
        'base_lex_consider_support',  # base + support
        'base_lex_consider_nag',  # base + negation
        'base_light_lex',  # base + light
        'base_light_lex_very',  # base + light + very
        'base_light_lex_consider_nag_very',  # base + light + very + negation
        'all_levels'  # based + light + very + emoji + support + negation
    )
    for method in methods:
        outfile = sys.argv[0][:-2] + '_' + method + '_' + '.result'
        sys.stdout = open(outfile, mode='w', encoding='utf-8')
        corpus = load_tweets(corpus_dir='./corpus_nora/corpus/')
        y_pred = do_sa_lex(tweets_corpus=corpus, method=method)
        y_true = [label for text, label in corpus]
        cm = ConfusionMatrix(actual_vector=y_true,
                             predict_vector=y_pred)  # Create CM From Data

        print('----------- summary results for hard voting -----------------')
        print('classes:\n', cm.classes)
        print('ACC(Accuracy)', cm.class_stat.get('ACC'))
        print('F1 score', cm.class_stat.get('F1'))
        print('Accuracy AVG', sum(cm.class_stat.get('ACC').values()) / len(cm.class_stat.get('ACC')))
        print('F1 AVG', sum(cm.class_stat.get('F1').values()) / len(cm.class_stat.get('F1')))
        print('----------------------------------------------')
    # for method in methods:
    #     do_sa_lex(corpus_dir='./corpus_Aldayel/split', method=method)
