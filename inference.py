#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn import metrics
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from conf import settings
from utils import get_network, get_test_dataloader
from plot_auc import my_metrics_plot

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-test-file', type=str, default='./data/test.txt', help='batch size for dataloader')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-plot-auc', type=bool, default=True, help='plot auc image')
    parser.add_argument('-gen-output', type=bool, default=True, help='generate output')
    args = parser.parse_args()
    TRAIN_MEAN, TRAIN_STD = [0.12206179, 0.09673566, 0.20968879], [0.12873057, 0.13296286, 0.2005473]
    net = get_network(args)

    test_loader = get_test_dataloader(
        TRAIN_MEAN,
        TRAIN_STD,
        args.test_file,
        num_workers=4,
        batch_size=args.b,
    )

    net.load_state_dict(torch.load(args.weights))
    #print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    pred_softmax_list = []
    label_list = []

    with torch.no_grad():
        for n_iter, (image, label) in tqdm(enumerate(test_loader)):
            #print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                #print('GPU INFO.....')
                #print(torch.cuda.memory_summary(), end='')


            output = net(image)
            output_softmax = nn.Softmax(dim=1)(output)
            #_, pred = output.topk(5, 1, largest=True, sorted=True)
            _, pred = output.topk(2, 1, largest=True, sorted=True)
            #print(pred.cpu().detach().numpy()[0].astype(int)[0])
            pred_softmax = output_softmax[0][:2]
            pred_softmax_list.append(pred_softmax.cpu().detach().numpy())
            #print(pred_softmax.cpu().detach().numpy())

            #label = label.view(label.size(0), -1).expand_as(pred)
            label_ = label.view(label.size(0), -1)
            label = label.view(label.size(0), -1).expand_as(pred)
            label_list.append(label_.cpu().detach().numpy())
            correct = pred.eq(label).float()

            #compute top 5
            #correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    #if args.gpu:
    #    print('GPU INFO.....')
    #    print(torch.cuda.memory_summary(), end='')

    #print(np.squeeze(np.array(label_list), axis=1).shape)
    #print(np.array(pred_softmax_list).shape)
    label_list = np.squeeze(np.array(label_list), axis=1)
    pred_softmax_list = np.array(pred_softmax_list)
    net_name = args.net
    out_path = 'model_auc'
    if args.gen_output:
        if not os.path.exists('./out_results'):
            os.makedirs('./out_results')
        df = pd.DataFrame(np.array([label_list.reshape(-1), pred_softmax_list[:,1].reshape(-1)]).T, columns=['{}-label'.format(args.net), '{}-pred'.format(args.net)])
        df.to_csv('./out_results/{}.csv'.format(args.net))
    if args.plot_auc:
        roc_auc, sensitivity, specificity, precision, accuracy, dice = my_metrics_plot(label_list, pred_softmax_list[:,1], net_name, out_path)
        print('roc_auc:{}, sensitivity:{}, specificity:{}, precision:{}, accuracy:{}, dice:{}'.format(roc_auc, sensitivity, specificity, precision, accuracy, dice))
    #fpr, tpr, thresholds = metrics.roc_curve(label_list, pred_softmax_list[:,1], pos_label=1)
    #auc = metrics.auc(fpr, tpr)
    print()
    #print('ROC_AUC:{}'.format(roc_auc))
    #print('AUC:{}'.format(auc))
    print("Top 1 err: ", 1 - correct_1 / len(test_loader.dataset))
    print("Top 1 acc: ", correct_1 / len(test_loader.dataset))
    #print("Top 5 err: ", 1 - correct_5 / len(test_loader.dataset))
    #print("Top 5 acc: ", correct_5 / len(test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
