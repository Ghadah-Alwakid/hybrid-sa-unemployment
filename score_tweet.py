from lexi_utilties import *

# load lexicons
# positive lexicon

lex_dir = './lexicons/lexicons_v7/'
print('lexicon dir', lex_dir)
pos_lex = load_lex(lex_dir + 'pos.txt')
pos_lex_light = load_lex(lex_dir + 'light_pos.txt')
pos_lex_emoji = load_lex(lex_dir + 'emoji_pos.txt')
# very positive lexicon
v_pos_lex = load_lex(lex_dir + 'very_pos.txt')
v_pos_lex_light = load_lex(lex_dir + 'light_very_pos.txt')
v_pos_lex_emoji = load_lex(lex_dir + 'emoji_very_pos.txt')
# ---------------------------------------
# negative lexicon
neg_lex = load_lex(lex_dir + 'neg.txt')
neg_lex_light = load_lex(lex_dir + 'light_neg.txt')
neg_lex_emoji = load_lex(lex_dir + 'emoji_neg.txt')
# very negative lexicon
v_neg_lex = load_lex(lex_dir + 'very_neg.txt')
v_neg_lex_light = load_lex(lex_dir + 'light_very_neg.txt')
v_neg_lex_emoji = load_lex(lex_dir + 'emoji_very_neg.txt')
# ---------------------------------------
# positive phrases
pos_phrases = load_lex(lex_dir + 'phrases_pos.txt')
# ---------------------------------------
# load negation list
negation_list = load_lex(lex_dir + 'negations.txt')
print('negation list:', negation_list)
# ---------------------------------------
# load support list (very, extremely, high .....)
support_list = load_lex(lex_dir + 'support_words.txt')
print('support list:', support_list)

# special lexicon
lex_dir = lex_dir + 'special_lex/'
special_lex = load_lex_dir(lex_dir)
print('special lexicon:', special_lex)
# ---------------------------------------
# load features
feature_dir = 'my_features_v2/'
features_dict = load_features(feature_dir)


####################################################
# scores
####################################################


# base lexicon (pos and neg)
def get_score_base_lex(word):
    # positive
    if word in pos_lex:
        score = 0.5
        sent_word = word
    # negative
    elif word in neg_lex:
        score = -0.5
        sent_word = word
    else:  # neutral
        score = 0.0
        sent_word = ''
    return score, sent_word


###################################################
###################################################
# emoji lexicon pos and neg emoji
def get_score_emoji(word):
    my_pos_lex = pos_lex_emoji
    my_neg_lex = neg_lex_emoji
    # positive
    if word in my_pos_lex:
        score = 0.5
        sent_word = word
    # negative
    elif word in my_neg_lex:
        score = -0.5
        sent_word = word
    else:  # neutral
        score = 0.0
        sent_word = ''
    return score, sent_word


###################################################
# base lexicon (pos and neg lexicons + pos and neg emoji)
def get_score_base_emoji(word):
    # sent_word = None
    # score = 0.0
    my_pos_lex = pos_lex + pos_lex_emoji
    my_neg_lex = neg_lex + neg_lex_emoji
    # positive
    if word in my_pos_lex:
        score = 0.5
        sent_word = word
    # negative
    elif word in my_neg_lex:
        score = -0.5
        sent_word = word
    else:  # neutral
        score = 0.0
        sent_word = ''

    return score, sent_word


###################################################
# base lexicons +/-, very +/-
def get_score_very_lex(word):
    my_pos_lex = pos_lex
    my_v_pos_lex = v_pos_lex
    my_neg_lex = neg_lex
    my_v_neg_lex = v_neg_lex
    # positive
    if word in my_pos_lex:
        score = 0.5
        sent_word = word
    # very positive
    elif word in my_v_pos_lex:
        score = 1.0
        sent_word = word
    # negative
    elif word in my_neg_lex:
        score = -0.5
        sent_word = word
    # very negative
    elif word in my_v_neg_lex:
        score = -1.0
        sent_word = word
    else:  # neutral
        score = 0.0
        sent_word = ''

    return score, sent_word


###################################################
# base lexicons +/-, very +/-, +/- emoji, very +/- and emoji
def get_score_very_lex_emoji(word):
    # sent_word = None
    # score = 0.0
    my_pos_lex = pos_lex + pos_lex_emoji
    my_v_pos_lex = v_pos_lex + v_pos_lex_emoji
    my_neg_lex = neg_lex + neg_lex_emoji
    my_v_neg_lex = v_neg_lex + v_neg_lex_emoji
    # positive
    sent_word = None
    if word in my_pos_lex:
        score = 0.5
        sent_word = word
    # very positive
    elif word in my_v_pos_lex:
        score = 1.0
        sent_word = word
    # negative
    elif word in my_neg_lex:
        score = -0.5
        sent_word = word
    # very negative
    elif word in my_v_neg_lex:
        score = -1.0
        sent_word = word
    else:  # neutral
        score = 0.0
        sent_word = ''
    return score, sent_word


###################################################
# base lexicon and consider support
def get_score_base_lex_consider_support(prev_word, word, next_word):
    # positive
    if word in pos_lex:
        score = 0.5
        sent_word = word
        is_support, support_phrase = contains_support(prev_word, word, next_word)
        if is_support:
            score += 0.5
            sent_word = support_phrase
    # negative
    elif word in neg_lex:
        score = -0.5
        sent_word = word
        is_support, support_phrase = contains_support(prev_word, word, next_word)
        if is_support:
            score -= 0.5
            sent_word = support_phrase
    else:  # neutral
        score = 0.0
        sent_word = ''
    return score, sent_word


#############################################
# base lexicon and consider negation
def get_score_base_lex_consider_nag(prev_word, word):
    # positive
    my_pos_lex = pos_lex
    my_neg_lex = neg_lex
    if word in my_pos_lex:
        if prev_word in negation_list:
            score = -0.5
            sent_word = prev_word + ' ' + word
        else:
            score = 0.5
            sent_word = word
    # negative
    elif word in my_neg_lex:
        if prev_word not in negation_list:
            score = 0.5
            sent_word = prev_word + ' ' + word
        else:
            score = -0.5
            sent_word = word
    else:  # neutral
        score = 0.0
        sent_word = ''

    return score, sent_word


#############################################


###################################################
# base and light lexicon
def get_score_base_light_lex(word, light_word):
    # sent_word = None
    # score = 0.0
    my_pos_lex = pos_lex + pos_lex_light
    my_neg_lex = neg_lex + neg_lex_light
    # positive
    if word in my_pos_lex or light_word in my_pos_lex:
        score = 0.5
        sent_word = word
    # negative
    elif word in my_neg_lex or light_word in my_neg_lex:
        score = -0.5
        sent_word = word
    else:  # neutral
        score = 0.0
        sent_word = ''

    return score, sent_word


###################################################


###################################################
# base + light + very
def get_score_base_light_lex_very(word, light_word):
    # sent_word = None
    # score = 0.0
    my_pos_lex = pos_lex + pos_lex_light
    my_v_pos_lex = v_pos_lex + v_pos_lex_light
    my_neg_lex = neg_lex + neg_lex_light
    my_v_neg_lex = v_neg_lex + v_neg_lex_light
    # positive
    if word in my_pos_lex or light_word in my_pos_lex:
        score = 0.5
        sent_word = word
    # very positive
    elif word in my_v_pos_lex or light_word in my_v_pos_lex:
        score = 1.0
        sent_word = word
    # negative
    elif word in my_neg_lex or light_word in my_neg_lex:
        score = -0.5
        sent_word = word
    # very positive
    elif word in my_v_neg_lex or light_word in my_v_neg_lex:
        score = -1.0
        sent_word = word
    else:  # neutral
        score = 0.0
        sent_word = ''

    return score, sent_word


###################################################


#############################################
# consider negation + light lexicon + very
def get_score_base_light_lex_consider_nag_very(prev_word, word, light_word):
    sent_word = None
    # score = 0.0
    # positive
    if word in pos_lex or light_word in pos_lex:
        if prev_word not in negation_list:
            score = 0.5
            sent_word = word
        else:
            score = -0.5
            sent_word = prev_word + ' ' + word
    # negative
    elif word in neg_lex or light_word in neg_lex:
        if prev_word not in negation_list:
            score = -0.5
            sent_word = word
        else:
            score = 0.5
            sent_word = prev_word + ' ' + word
    # very positive
    elif word in v_pos_lex or light_word in v_pos_lex:
        score = 1.0
    # very negative
    elif word in v_neg_lex or light_word in v_neg_lex:
        score = -1.0
    else:  # neutral
        score = 0.0
        sent_word = ''

    return score, sent_word


#############################################


# to consider negation and support
def get_score_all_levels(prev_word, word, next_word, light_word):
    # sent_word = None
    # score = 0.0
    my_pos_lex = pos_lex + pos_lex_light + pos_lex_emoji
    my_v_pos_lex = v_pos_lex + v_pos_lex_light + v_pos_lex_emoji
    my_neg_lex = neg_lex + neg_lex_light + neg_lex_emoji
    my_v_neg_lex = v_neg_lex + v_neg_lex_light + v_neg_lex_emoji
    # positive
    if word in my_pos_lex or light_word in my_pos_lex:
        # consider negation
        if prev_word in negation_list:
            score = -0.5
            sent_word = prev_word + ' ' + word
        else:
            score = 0.5
            sent_word = word
        # contains support in positive
        is_support, support_phrase = contains_support(prev_word, word, next_word)
        if is_support:
            score += 0.5
            sent_word = support_phrase
    # very positive
    elif word in my_v_pos_lex or light_word in my_v_pos_lex:
        if prev_word in negation_list:
            score = -1.0
            sent_word = prev_word + ' ' + word
        else:
            score = 1.0
            sent_word = word
        # contains support in very positive
        is_support, support_phrase = contains_support(prev_word, word, next_word)
        if is_support:
            score += 0.5
            sent_word = support_phrase
    # negative
    elif word in my_neg_lex or light_word in my_neg_lex:
        if prev_word in negation_list:
            score = 0.5
            sent_word = prev_word + ' ' + word
        else:
            score = -0.5
            sent_word = word
        # contains support in negative
        is_support, support_phrase = contains_support(prev_word, word, next_word)
        if is_support:
            score -= 0.5
            sent_word = support_phrase
    # very negative
    elif word in my_v_neg_lex or light_word in my_v_neg_lex:
        if prev_word in negation_list:
            score = 1.0
            sent_word = prev_word + ' ' + word
        else:
            score = -1.0
            sent_word = word
        # contains support in very negative
        is_support, support_phrase = contains_support(prev_word, word, next_word)
        if is_support:
            score -= 0.5
            sent_word = support_phrase
    else:  # neutral
        score = 0.0
        sent_word = ''

    return score, sent_word


#############################################


def contains_support(prev_word, word, next_word):
    if prev_word in support_list:
        return True, prev_word + ' ' + word
    elif next_word in support_list:
        return True, word + ' ' + next_word
    else:
        return False, None


'''
Rules 
if the tweet contain special_lex ==> Negative directly
if the tweet contain phrases-pos ==> Positive directly
'''


def contain_special_lex(tweet):
    for phrase in special_lex:
        if phrase in tweet:
            return True
    return False


def contain_pos_phrase(tweet):
    for phrase in pos_phrases:
        if phrase in tweet:
            return True
    return False
