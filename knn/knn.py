import numpy as np
import pandas as pd
import os
import sys
import subprocess
from sklearn.metrics import classification_report

N_CLASSES = 2
target_names = ['class {}'.format(class_id) for class_id in range(N_CLASSES)]

class KNN_Classifier:
    def __init__(self, train_points, train_labels, k=10, metric='euclidean'):
        self.train_points = train_points
        self.train_labels = train_labels
        self.k = k
        self.metric = metric
        self.N_train, self.M = train_points.shape

    

    def knn(self, test_points):
        N_test, M = test_points.shape
        pp_distance = np.full(N_test, np.inf)
        if self.metric == 'absolute':
            pp_distance = np.array([[np.sum(np.abs(x - y)) for y in self.train_points] for x in test_points])
        elif self.metric == 'cosine':
            # AB/(||A|| ||B||)
            dotroducts = np.array([[np.sum(x*y) for y in self.train_points] for x in test_points])
            inv_test_norms = 1.0 / np.sqrt(np.array([[np.sum(x*x) for y in self.train_points] for x in test_points]))
            inv_train_norms = 1.0 / np.sqrt(np.array([[np.sum(y*y) for y in self.train_points] for x in test_points]))
            pp_distance = dotroducts * inv_test_norms * inv_train_norms
        elif self.metric == 'euclidean':
            pp_distance = np.sqrt(np.array([[np.sum(y*y) for y in self.train_points] for x in test_points]))
        else:
            print("Unknown metric.  The default Euclidean metric will be used.")
            pp_distance = np.sqrt(np.array([[np.sum(y*y) for y in self.train_points] for x in test_points]))
        return np.argsort(pp_distance, axis=1)[:, 0:self.k]
    


    def classify(self, test_points):
        # N_test, M = test_points.shape
        # N_test
        pred_labels = np.array([self.train_labels[x] for x in self.knn(test_points)], dtype=int)
        majority_vote = [np.argmax(np.bincount(x)) for x in pred_labels]
        return majority_vote


def collect_data_by_params(n, m, fold_id):
    df = pd.read_csv("data/n{}m{}train{}.csv".format(n, m, fold_id))
    train_data = df.to_numpy()[:,:-1]
    train_target = df.to_numpy()[:,-1]
    df = pd.read_csv("data/n{}m{}test{}.csv".format(n, m, fold_id))
    test_data = df.to_numpy()[:,:-1]
    test_target = df.to_numpy()[:,-1]
    return train_data, train_target, test_data, test_target

def save_results_by_params(n, m, fold_id, y_true, y_pred):
    df = pd.read_csv("data/n{}m{}test{}.csv".format(n, m, fold_id))
    df['label'] = y_pred
    df.to_csv("results/n{}m{}pred{}.csv".format(n, m, fold_id), index=False)
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv("results/n{}m{}report{}.csv".format(n, m, fold_id))


MAX_N = 1.5e4
N_NUM = 3
MAX_M = 40
M_NUM = 5
N_CLASSES = 2

bashCommand = "rm -rf results"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
os.mkdir('results')

n_list = np.linspace(50, MAX_N, num=N_NUM, dtype=int)
m_list = np.linspace(2, MAX_M, num=M_NUM, dtype=int)
fold_list = range(4)

for n in n_list*N_CLASSES:
    for m in m_list:
        for fold_id in fold_list:
            x_train, y_train, x, y_true = collect_data_by_params(n, m, fold_id)
            classifier = KNN_Classifier(x_train, y_train, metric='absolute')
            y_pred = classifier.classify(x)
            save_results_by_params(n, m, fold_id, y_true, y_pred)
