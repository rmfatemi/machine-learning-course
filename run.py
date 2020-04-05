__author__ = "Rasool Fatemi"
__version__ = "0.1"
__email__ = "rmfatemi@gmail.com"

import pandas as pd
import numpy as np
import sys
from algorithms import *

# Only used to change categorical values to numerical ones (line 40)
from sklearn.preprocessing import LabelEncoder


def repeated_kfold(data, n_folds=3, n_repeats=1):
    dataset = []

    for i in range(n_repeats):
        np.random.shuffle(data)
        chunks = np.array_split(data, n_folds)
        for j in range(n_folds):
            copy = chunks
            test = chunks[j]
            chunks = np.delete(chunks, j)
            train = np.vstack(chunks)
            dataset.append([train, test])
            chunks = copy
    return np.array(dataset)


def accuracy_score(true_val, pred_val):
    return (sum(true_val == pred_val) / len(true_val) * 100.0)


def get_raw_data(name):
    raw_data = pd.read_csv(name)
    raw_data = raw_data.apply(LabelEncoder().fit_transform)
    raw_data = np.array(raw_data)
    return raw_data


def split_data_labels(name, raw_data):
    if name == 'letter-recognition.data' or name == 'mushroom.data':
        data = raw_data[:, 1:]
        labels = raw_data[:, 0]
    else:
        data = raw_data[:, 0:raw_data.shape[1] - 1]
        labels = raw_data[:, raw_data.shape[1] - 1]
    return data, labels


if __name__ == '__main__':

    name = str(sys.argv[-1])
    raw_data = get_raw_data(name)

    scores = []
    rkf = repeated_kfold(raw_data, n_folds=5, n_repeats=10)

    for data in rkf:
        train, test = data
        train_data, train_labels = split_data_labels(name, train)
        test_data, test_labels = split_data_labels(name, test)

        # Before running comment out all algorithms except the one you want to run

        # 'DECISION TREE'
        attributes = list(range(len(train_data[0])))
        clf = DecisionTree(train_data, train_labels, attributes, max_depth=5)
        pred = clf.classify(test_data)

        # # 'ADABOOST ON STUMPS'
        # clf = Adaboost()
        # attributes = list(range(len(train_data[0])))
        # pred = clf.classify(train_data, train_labels, test_data, attributes, M=5)

        # # 'RANDOM FOREST'
        # clf = RandomForest(60, 100, 2)
        # pred = clf.classify(train_data, train_labels, test_data)

        # 'k-NEAREST NEIGHBORS'
        # clf = KNN()
        # pred = clf.classify(train_data, train_labels, test_data, test_labels, k=9)

        # # 'NAIVE BAYES'
        # clf = NaiveBayes()
        # pred = clf.classify(train_data, train_labels, test_data, test_labels)

        scores.append(accuracy_score(test_labels, pred))
        print(accuracy_score(test_labels, pred))

    std = round(np.std(scores), 2)
    acc = round(np.average(scores), 2)
    print('ACC: ', acc, ' STD: ', std)
