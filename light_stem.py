from textutil import *


def do_work(infile_name, outfile_name):
    print('in file', infile_name)
    print('out file', outfile_name)
    infile = open(infile_name, encoding='utf-8')
    outfile = open(outfile_name, mode='w', encoding='utf-8')
    for line in infile:
        if not line.strip():
            continue
        for word in line.split():
            light_word = light_stem_word(word)
            outfile.write(light_word + ' ')
        outfile.write('\n')
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
