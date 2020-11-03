import re
from itertools import islice
from nltk import ISRIStemmer
import aspell

ar_spell = aspell.Speller(('dict-dir', './ar_dict/'), ('lang', 'ar'),
                          ('encoding', 'utf-8'))


def window(words_seq, n):
    """Returns a sliding window (of width n) over data from the iterable"""
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(words_seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def remove_repeating_char(text):
    # return re.sub(r'(.)\1+', r'\1', text)     # keep only 1 repeat
    return re.sub(r'(.)\1+', r'\1\1', text)  # keep 2 repeat


def process_text(text, n=1,
                 remove_vowel_marks=False,
                 remove_repeated_chars=False,
                 ):
    clean_text = text
    if remove_vowel_marks:
        clean_text = remove_diacritics(clean_text)
    if remove_repeated_chars:
        clean_text = remove_repeating_char(clean_text)

    if n == 1:
        return clean_text.split()
    else:
        tokens = clean_text.split()
        grams = tokens
        for i in range(2, n + 1):
            grams = list(window(tokens, i))
            grams = [' '.join(g) for g in grams]
        return grams


arabic_diacritics = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)


def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text


def load_data(data_file, label, n):
    print('read {} data ...'.format(label))
    data_features = list()
    data = list()
    infile = open(data_file, encoding='utf-8')
    for line in infile:
        if not line.strip():
            continue
        text_features = process_text(line, n)
        data_features += text_features
        data.append((text_features, label))
    return data, data_features


def document_features(document, corpus_features):
    document_words = set(document)
    features = {}
    for word in corpus_features:
        features['has({})'.format(word)] = (word in document_words)
    return features


def read_data(data_file, label):
    # print('read {} data ...'.format(label))
    text_data = list()
    labels = list()
    infile = open(data_file, encoding='utf-8')
    for line in infile:
        if not line.strip():
            continue
        text_data.append(line)
        labels.append(label)
    return text_data, labels


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
