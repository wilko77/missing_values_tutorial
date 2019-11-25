import pickle
import os
from sklearn import metrics
from matplotlib import pyplot as plt


def compute_auc_and_f1(precisions, recalls):
    precisions.insert(0, 1.0)
    recalls.insert(0, 0.0)
    precisions.append(0.0)
    recalls.append(1.0)
    auc = metrics.auc(recalls, precisions)

    def f1(prec, recall):
        try:
            res = 2 * prec * recall / (prec + recall)
        except ZeroDivisionError:
            res = 0
        return res

    best_f1 = max(map(f1, precisions, recalls))
    return auc, best_f1


def get_aucs(folder, strategies):
    x = range(0, 81, 5)
    aucs = {s: [] for s in strategies}
    for i in x:
        for strategy in strategies:
            with open(os.path.join(folder, f'{i:02}_pr_{strategy}.pkl'), 'rb') as f:
                precisions, recalls = pickle.load(f)
                auc, best_f1 = compute_auc_and_f1(precisions, recalls)
                aucs[strategy].append(auc)
    return aucs


def print_auc_and_f1(folder, strategies):
    x = range(0, 81, 5)
    aucs = get_aucs(folder, strategies)
    for k, v in aucs.items():
        plt.plot(x, v, label=k)
    plt.title('baseline with tokenized comparisons - noisefree')
    plt.legend()
    plt.xlabel('% missing data')
    plt.ylabel('area under PR curve')
    plt.show()
