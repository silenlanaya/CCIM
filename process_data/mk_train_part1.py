import os
from tqdm import tqdm
import pandas as pd
import random


data_path = '/home/wangyin/data/respone_part1.xlsx'
train_dst = '../data/train.txt'
test_dst = '../data/test.txt'
df = pd.read_excel(io=data_path)
names = df['sampleID'].values.tolist()
labels = df['outcome'].values.tolist()
imgs = os.listdir('/home/wangyin/data/mIHC')
train_ratio = 0.8


dataset = []
with open(train_dst, 'w') as ftr:
    with open(test_dst, 'w') as ftt:
        for img in imgs:
            head = img.split('_')[0]
            for name, label in zip(names, labels):
                if name == head:
                    #print(img, label)
                    dataset.append([img, label])
        random.shuffle(dataset)
        train_set = dataset[:int(len(dataset)*train_ratio)]
        test_set = dataset[int(len(dataset)*train_ratio):]
        for img, label in train_set:
            ftr.writelines(img + ',' + str(label) + '\n')
        for img, label in test_set:
            ftt.writelines(img + ',' + str(label) + '\n')

