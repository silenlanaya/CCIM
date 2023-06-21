import os
from tqdm import tqdm
import pandas as pd
import random



train_dst = '../data/train.txt'
test_dst = '../data/test.txt'
part2 = '../data/part2.txt'
train_ratio = 0.8

fp2 = open(part2, 'r')
part2_lines = fp2.readlines()

with open(train_dst, 'a') as ftr:
    with open(test_dst, 'a') as ftt:
        random.shuffle(part2_lines)
        ftr.writelines(part2_lines[:int(len(part2_lines)*train_ratio)])
        ftt.writelines(part2_lines[int(len(part2_lines)*train_ratio):])
fp2.close()
