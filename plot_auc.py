import numpy as np
import os
import time
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
np.seterr(divide='ignore',invalid='ignore')


def my_metrics_plot(gt, pred, net_name, out_path):
    fpr, tpr, thresholds = roc_curve(gt, pred, pos_label=1)
    if np.isnan(fpr).all():
        fpr = np.array([0, 0, 1])
    if np.isnan(tpr).all():
        tpr = np.array([0, 0, 1])
    #print('\nfpr, tpr:{}, {}'.format(fpr, tpr))
    pred_binary = (pred > 0.5).astype(np.int_)
    tn, fp, fn, tp = confusion_matrix(gt.tolist(), pred_binary.tolist(), labels=[0, 1]).ravel()
    #print('tp:{}, fp:{}'.format(tp, fp))
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    dice = 2*tp / (fp + 2*tp + fn + 1e-8)
    #tpr = tp / (tp + fn + 1e-8)
    #fpr = fp / (fp + tn + 1e-8)
    roc_auc = metrics.auc(fpr, tpr)
    #print('roc_auc:{}'.format(roc_auc))
    lw = 2
    ax = plt.subplot(111)
    ax.plot(fpr, tpr, color='black',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    ax.plot([0, 1], [0, 1], color='darkred', lw=lw)
    ax.set_aspect('equal', adjustable='box')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    #plt.title('Receiver operating characteristic example', fontsize=14)
    plt.legend(loc="lower right", fontsize=14)
    plt.grid()
    #plt.savefig('./outputs_single/msi_auc_201506139.png')
    plt.savefig('./{}/{}-{}.png'.format(out_path, net_name, time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))))
    return roc_auc, sensitivity, specificity, precision, accuracy, dice
