import operator
import os
from glob import glob
from itertools import islice

import aspell
from nltk import ISRIStemmer

# suspended because of incompatibility error
# aspell.AspellSpellerError: The file "./ar_dict//ar.rws" is not in the proper format. Incompatible hash function.

ar_spell = aspell.Speller(('dict-dir', './ar_dict'), ('lang', 'ar'),
                          ('encoding', 'utf-8'))


def get_label(pos_score, neg_score):
    if pos_score + neg_score > 0:
        return 'pos'
    if pos_score + neg_score <= 0:
        return 'neg'
    if pos_score == 0 and neg_score == 0:
        return 'pos'


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def load_lex(file_name):
    infile = open(file_name, encoding='utf-8')
    lines = infile.read().split('\n')
    lexicon = [token for token in lines if not token.startswith('#')]
    infile.close()
    return lexicon


# suspended because of incompatibility error
def separate_waw(word):
    if word.startswith('و'):
        root = get_root_word(word[1:])
        # print(root)
        if root not in ar_spell:
            return word
        else:
            return 'و ' + word[1:]
    else:
        return word


def load_corpus(corpus_file, sep=True):
    lines = open(corpus_file, encoding='utf-8').read().split('\n')
    if sep is False:
        return [line for line in lines if line.strip()]
    clean_lines = list()
    # separate waw from words
    for line in lines:
        if not line.strip():
            continue
        sentence = ''
        for word in line.split():
            if word.startswith('و'):
                sentence += separate_waw(word) + ' '
            else:
                sentence += word + ' '
        clean_lines.append(sentence)
    return clean_lines


def load_lex_dir(lex_directory):
    files = glob(lex_directory + '*.txt')
    lex = list()
    for lex_file in files:
        lex.extend(load_lex(lex_file))
    return lex


def load_features(feature_directory):
    features_dictionary = {}
    for filename in os.listdir(feature_directory):
        if filename.endswith(".lst") or filename.endswith(".txt"):
            # print('processing file', filename)
            feature_file = os.path.join(feature_directory, filename)
            entities = load_lex(feature_file)
            entity_name, ext = filename.split('.')
            features_dictionary[entity_name] = entities
    return features_dictionary


def light_stem_word(word):
    original_word = word
    arabic_stemmer = ISRIStemmer()
    # remove diacritics which representing Arabic short vowels
    word = arabic_stemmer.norm(word, num=1)
    # exclude stop words from being processed
    if word not in arabic_stemmer.stop_words:
        # remove length three and length two prefixes in this order
        word = arabic_stemmer.pre32(word)
        # remove length three and length two suffixes in this order
        word = arabic_stemmer.suf32(word)
        # remove connective ‘و’ if it precedes a word beginning with ‘و’
        word = arabic_stemmer.waw(word)
        # normalize initial hamza to bare alif
        word = arabic_stemmer.norm(word, num=2)
    if word not in ar_spell:
        return original_word
    else:
        return word


def light_stem(text):
    if not isinstance(text, list):
        word_list = text.split()
    else:
        word_list = [text]
    result = list()
    for word in word_list:
        light_word = light_stem_word(word)
        result.append(light_word)
    return result


def get_root_word(word):
    arabic_stemmer = ISRIStemmer()
    root = arabic_stemmer.stem(word)
    return root


def get_root(text):
    arabic_stemmer = ISRIStemmer()
    if not isinstance(text, list):
        word_list = text.split()
    else:
        word_list = text
    result = list()
    for word in text:
        root = arabic_stemmer.stem(word)
        result.append(root)
    return result


def most_frequent(word_list, n=10):
    word_freq = {}
    for word in word_list:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    sorted_freq = sorted(word_freq.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_freq[:n]


def get_feature_from_word(word, features_dict):
    for feature_name, feature_list in features_dict.items():
        if word in feature_list:
            return feature_name
    return None


def find_features_in_text(text, features_dict):
    feature_set = set()
    for feature_name, feature_list in features_dict.items():
        for feature in feature_list:
            if feature in text:
                feature_set.add((feature, feature_name))
    return feature_set


def contains_negation(tweet, negation_list):
    words = tweet.split()
    for word in words:
        if word in negation_list:
            return True
    return False


def contains_support_words(tweet, support_list):
    words = tweet.split()
    for word in words:
        if word in support_list:
            return True
    return False


def print_sentiment_score(sentiment, score):
    print('sentiment: {}\tscore: {}'.format(sentiment, score))
