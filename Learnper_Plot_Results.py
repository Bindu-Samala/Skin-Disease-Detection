import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
#from Test_AUC import AUC


def plot_results():
    # matplotlib.use('TkAgg')
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Term = ['TP', 'TN', 'FP', 'FN']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Classifier = ['TERMS', 'CNN', 'Resnet', 'Mobilenet', 'VGG16', 'Proposed']
    value = eval[4, :, 4:]

    value1 = eval[4, :, 0:4]

    Table = PrettyTable()
    Table.add_column(Classifier[0], Term)
    for j in range(len(Classifier) - 1):
        Table.add_column(Classifier[j + 1], value1[j, :])
    print('------------------------------ Classifier Comparison - Confusion Matrix',
          '-------------------------------')
    print(Table)

    Table = PrettyTable()
    Table.add_column(Classifier[0], Terms)
    for j in range(len(Classifier) - 1):
        Table.add_column(Classifier[j + 1], value[j, :])
    print('--------------------------------------------------------- Classifier Comparison',
          '---------------------------------------------------------')
    print(Table)

    learnper = [35, 45, 55, 65, 75, 85]
    for j in range(len(Terms)):
        Graph = np.zeros((eval.shape[0], eval.shape[1]))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                if j == 9:
                    Graph[k, l] = eval[k, l, j + 4]
                else:
                    Graph[k, l] = eval[k, l, j + 4] * 100

        # plt.plot(learnper, Graph[:, 0], color='r', linewidth=3, marker='o', markerfacecolor='blue', markersize=12,
        #          label="CNN")
        # plt.plot(learnper, Graph[:, 1], color='g', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
        #          label="RESNET")
        # plt.plot(learnper, Graph[:, 2], color='b', linewidth=3, marker='o', markerfacecolor='green', markersize=12,
        #          label="MOBILENET")
        # plt.plot(learnper, Graph[:, 3], color='m', linewidth=3, marker='o', markerfacecolor='yellow', markersize=12,
        #          label="VGG16")
        # plt.plot(learnper, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='cyan', markersize=12,
        #          label="MOBILENET-VGG16")
        # plt.xlabel('Learning Percentage (%)')
        # plt.ylabel(Terms[Graph_Term[j]])
        # plt.legend(loc='best')
        # path1 = "./Results/LineGraph_%s.png" % (Terms[Graph_Term[j]])
        # plt.savefig(path1)
        # plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(6)
        ax.bar(X + 0.00, Graph[:, 0], color='r', width=0.10, label="CNN 23]")
        ax.bar(X + 0.10, Graph[:, 1], color='g', width=0.10, label="ResNet [24]")
        ax.bar(X + 0.20, Graph[:, 2], color='b', width=0.10, label="Mobilenet [21]")
        ax.bar(X + 0.30, Graph[:, 3], color='m', width=0.10, label="VGG16 [22]")
        ax.bar(X + 0.40, Graph[:, 4], color='k', width=0.10, label="Mobilenet-VGG16")
        plt.xticks(X + 0.10, (35, 45, 55, 65, 75, 85))
        plt.xlabel('Learning Percentage (%)')
        plt.ylabel(Terms[Graph_Term[j]])
        plt.legend(loc=1)
        path1 = "./Results/BarGraph_%s.png" % (Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()



plot_results()
# AUC()