from statistics import mode

from jinja2.nodes import Mul
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier

from sklearn.svm import LinearSVC
from pycm import ConfusionMatrix

from sa_lex import *
from sa_ml_add_feature_sentiment_score import *

if __name__ == '__main__':
    my_data = load()
    x_train, y_train, x_test, y_test = my_data
    # y_predicted_ml1 = do_sa_ml(2, LinearSVC(), my_data)
    # y_predicted_ml1 = do_sa_ml(2, BernoulliNB(), my_data)
    y_predicted_ml1 = do_sa_ml(3, SGDClassifier(), my_data)
    cm = ConfusionMatrix(actual_vector=y_test,
                         predict_vector=y_predicted_ml1)  # Create CM From Data

    print('----------- summary results for hard voting -----------------')
    print('classes:\n', cm.classes)
    print('ACC(Accuracy)', cm.class_stat.get('ACC'))
    print('F1 score', cm.class_stat.get('F1'))
    print('Accuracy AVG', sum(cm.class_stat.get('ACC').values()) / len(cm.class_stat.get('ACC')))
    print('F1 AVG', sum(cm.class_stat.get('F1').values()) / len(cm.class_stat.get('F1')))
    print('----------------------------------------------')
