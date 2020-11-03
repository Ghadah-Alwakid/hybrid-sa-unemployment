from textutil import *
from lexi_utilties import *
from score_tweet import *


def plain_features(features):
    my_feat = ''
    for feat in features:
        my_feat = my_feat + feat[1] + ' '
    return my_feat


def do_work(infile_name, outfile_name):
    print('in file', infile_name)
    print('out file', outfile_name)
    infile = open(infile_name, encoding='utf-8')
    outfile = open(outfile_name, mode='w', encoding='utf-8')
    for line in infile:
        if not line.strip():
            continue
        features = find_features_in_text(line, features_dict)
        print(features)
        outfile.write(line.strip() + plain_features(features) + '\n')
    infile.close()
    outfile.close()


if __name__ == '__main__':
    do_work('corpus_v4/split/tweets_neg_test.txt',
            'corpus_v4/map/tweets_neg_test.txt')

    do_work('corpus_v4/split/tweets_pos_test.txt',
            'corpus_v4/map/tweets_pos_test.txt')

    do_work('corpus_v4/split/tweets_neg_train.txt',
            'corpus_v4/map/tweets_neg_train.txt')

    do_work('corpus_v4/split/tweets_pos_train.txt',
            'corpus_v4/map/tweets_pos_train.txt')

    print('all done')
