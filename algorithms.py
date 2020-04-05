__author__ = "Rasool Fatemi"
__version__ = "0.7"
__email__ = "rmfatemi@gmail.com"

import numpy as np
import random
import math
import scipy.stats as st


def entropy(attribute_data, weights):
    _, val_freqs = np.unique(attribute_data, return_counts=True)
    val_probs = val_freqs / len(attribute_data)
    return -val_probs.dot(np.log(val_probs))


def info_gain(attribute_data, labels, weights):
    attr_val_counts = get_count_dict(attribute_data)
    total_count = len(labels)
    EA = 0.0
    for attr_val, attr_val_count in attr_val_counts.items():
        EA += attr_val_count * entropy(labels[attribute_data == attr_val], weights[attribute_data == attr_val])

    return entropy(labels, weights) - EA / total_count


def get_count_dict(data):
    data_values, data_freqs = np.unique(data, return_counts=True)
    return dict(zip(data_values, data_freqs))


class DecisionTree:
    label = None
    attribute = None
    attribute_value = None
    children = None
    p_value = None
    parent = None
    level = None
    max_depth = 10000000
    weights = None

    def __init__(self, data, labels, attributes, info_gain=info_gain, value=None, parent=None, max_depth=None,
                 old_depth=0, weights=None):

        self.level = old_depth + 1

        if weights is not None:
            self.weights = weights

        if max_depth is not None:
            self.max_depth = max_depth

        if value is not None:
            self.attribute_value = value

        if parent is not None:
            self.parent = parent

        if data.size == 0 or not attributes or self.level == self.max_depth:
            try:
                self.label = st.mode(labels)[0][0]
            except:
                self.label = labels[len(labels) - 1]
            return

        if np.all(labels[:] == labels[0]):
            self.label = labels[0]
            return

        examples_all_same = True
        for i in range(1, data.shape[0]):
            for j in range(data.shape[1]):
                if data[0, j] != data[i, j]:
                    examples_all_same = False
                    break
            if not examples_all_same:
                break
        if examples_all_same:
            self.label = labels[len(labels) - 1]
            return

        self.build(data, labels, attributes, info_gain)
        return

    def build(self, data, labels, attributes, info_gain):

        self.choose_best_attribute(data, labels, attributes, info_gain)
        best_attribute_column = attributes.index(self.attribute)
        attribute_data = data[:, best_attribute_column]

        child_attributes = attributes[:]
        child_attributes.remove(self.attribute)

        self.children = []
        for val in np.unique(attribute_data):
            child_data = np.delete(data[attribute_data == val, :], best_attribute_column, 1)
            child_labels = labels[attribute_data == val]
            self.children.append(DecisionTree(child_data, child_labels, child_attributes, value=val, parent=self,
                                              old_depth=self.level, max_depth=self.max_depth))

    def choose_best_attribute(self, data, labels, attributes, info_gain):

        best_gain = float('-inf')
        for attribute in attributes:
            attribute_data = data[:, attributes.index(attribute)]
            if self.weights is None:
                gain = info_gain(attribute_data, labels, np.ones(len(attribute_data)))
            else:
                gain = info_gain(attribute_data, labels, self.weights)
            if gain > best_gain:
                best_gain = gain
                self.attribute = attribute
        return

    def classify(self, data):

        if data.size == 0:
            return

        if len(data.shape) == 1:
            data = np.reshape(data, (1, len(data)))

        if self.children is None:
            labels = np.ones(len(data)) * self.label
            return labels

        labels = np.zeros(len(data))

        for child in self.children:
            child_attr_val_idx = data[:, self.attribute] == child.attribute_value
            labels[child_attr_val_idx] = child.classify(data[child_attr_val_idx])

        return labels


class RandomForest:
    trees = None
    number_trees = None
    number_samples = None
    max_depth = None

    def __init__(self, number_trees, number_samples, max_depth=1):

        self.trees = []
        self.number_trees = number_trees
        self.number_samples = number_samples
        self.max_depth = max_depth

    def train_tree(self, train_data, train_labels):

        attributes = list(range(len(train_data[0])))
        tree = DecisionTree(train_data, train_labels, attributes, max_depth=self.max_depth)
        return tree

    def predict(self, test):

        pred = []

        for tree in self.trees:
            pred.append(tree.classify(test))

        for i in range(len(pred)):
            pred[i] = tuple(pred[i].tolist())

        return max(set(pred), key=pred.count)

    def classify(self, train_data, train_labels, test_data):

        rand_fts = []
        for i in range(self.number_trees):
            rand_fts.append([i, random.sample(list(train_data), self.number_samples)])

        for i in rand_fts:
            self.trees.append(self.train_tree(train_data, train_labels))

        pred = []
        for test in test_data:
            pred.append(self.predict(test))

        return self.predict(test_data)


class Adaboost:

    def classify(self, train_data, train_labels, test_data, attributes, M):
        n_train, n_test = len(train_data), len(test_data)
        w = np.ones(n_train) / n_train
        pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

        for i in range(M):
            clf = DecisionTree(train_data, train_labels, attributes, max_depth=1, weights=w)
            pred_train_i = clf.classify(train_data)
            pred_test_i = clf.classify(test_data)

            miss = [int(x) for x in (pred_train_i != train_labels)]
            miss2 = [x if x == 1 else -1 for x in miss]
            err_m = np.dot(w, miss) / sum(w)
            alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))
            w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
            pred_train = [sum(x) for x in zip(pred_train, [x * alpha_m for x in pred_train_i])]
            pred_test = [sum(x) for x in zip(pred_test, [x * alpha_m for x in pred_test_i])]

        pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
        return pred_test


class KNN:

    def dist(self, a, b):

        return abs(np.linalg.norm(b - a))

    def classify(self, train_data, train_labels, test_data, test_labels, k):

        pred = np.zeros(len(test_data))

        for test in test_data:
            i = 0
            distances = np.zeros((len(train_data), 2))
            for train in train_data:
                distances[i] = [train_labels[i], self.dist(test, train)]
                i += 1
            distances = distances[np.argsort(distances[:, 1])]
            distances = distances[:, 0].astype(int)

            neighbors = []
            for i in range(0, k):
                neighbors.append(distances[i])

            np.append(pred, np.bincount(neighbors).argmax())

        return pred


class NaiveBayes:

    def mean(self, numbers):

        return sum(numbers) / float(len(numbers))

    def std_dev(self, numbers):

        average = self.mean(numbers)
        variance = sum([pow(x - average, 2) for x in numbers]) / float(len(numbers) - 1)
        return math.sqrt(variance)

    def calculate_probability(self, x, mean, std_dev):

        prob = 0
        try:
            exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std_dev, 2))))
            prob = (1 / (math.sqrt(2 * math.pi) * std_dev)) * exponent
        except:
            pass
        return prob

    def summarize(self, train_data):

        summaries = [(self.mean(attribute), self.std_dev(attribute)) for attribute in zip(*train_data)]
        return summaries

    def separate_class(self, train_data, train_labels):

        separated = {}
        for data, label in zip(train_data, train_labels):
            if (label not in separated):
                separated[label] = []
            separated[label].append(data)
        return separated

    def summarize_class(self, train_data, train_labels):

        separated = self.separate_class(train_data, train_labels)
        summaries = {}
        for class_val, instances in separated.items():
            summaries[class_val] = self.summarize(instances)
        return summaries

    def calculate_class_probabilities(self, summaries, test):

        probabilities = {}
        for class_val, class_summaries in summaries.items():
            probabilities[class_val] = 1
            for i in range(len(class_summaries)):
                mean, std_dev = class_summaries[i]
                x = test[i]
                probabilities[class_val] *= self.calculate_probability(x, mean, std_dev)
        return probabilities

    def predict(self, summaries, test):

        probabilities = self.calculate_class_probabilities(summaries, test)
        best_label, best_prob = None, -1
        for class_val, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_val
        return best_label

    def classify(self, train_data, train_labels, test_data, test_labels):

        summaries = self.summarize_class(train_data, train_labels)

        pred = []
        for test in test_data:
            pred.append(self.predict(summaries, test))

        return pred
