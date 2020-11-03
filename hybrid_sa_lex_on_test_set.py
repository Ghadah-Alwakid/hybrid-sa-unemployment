from statistics import mode
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from pycm import ConfusionMatrix

from sa_lex import *
from sa_ml import *


def hard_voting(y1, y2, y3):
    answer = []
    for l1, l2, l3 in zip(y1, y2, y3):
        label = mode([l1, l2, l3])
        answer.append(label)
    return np.array(answer)


if __name__ == '__main__':
    my_data = load()
    x_train, y_train, x_test, y_test = my_data

    corpus = load_tweets(corpus_dir='./corpus_v4/split/')
    y_predicted_lex = do_sa_lex(tweets_corpus=corpus,
                                method='base_light_lex_consider_nag_very')
    y_predicted_lex = np.array(y_predicted_lex)
    print(len(y_test), len(y_predicted_lex))
    cm = ConfusionMatrix(actual_vector=y_test,
                         predict_vector=y_predicted_lex)  # Create CM From Data

    print('----------- summary results for hard voting -----------------')
    print('classes:\n', cm.classes)
    print('ACC(Accuracy)', cm.class_stat.get('ACC'))
    print('F1 score', cm.class_stat.get('F1'))
    print('Accuracy AVG', sum(cm.class_stat.get('ACC').values()) / len(cm.class_stat.get('ACC')))
    print('F1 AVG', sum(cm.class_stat.get('F1').values()) / len(cm.class_stat.get('F1')))
    print('----------------------------------------------')
