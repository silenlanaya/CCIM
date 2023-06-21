import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import shutil

part1_root = '/home/wangyin/data/response_part1.xlsx'
part2_root = '/home/wangyin/data/response_part2.xlsx'
part2_data_root = '/home/wangyin/data/mIHC_part2'
dst = '/home/wangyin/data/mIHC'
#dst = '/home/wangyin/data/part2_tif'
part2_dataset = '../data/part2.txt'

df1 = pd.read_excel(io=part1_root)
df2 = pd.read_excel(io=part2_root)
id1 = df1['sampleID']
id2 = df2['sampleID']
label2 = df2['outcome']
id1_str = [str(i) for i in id1]
id2_str = [str(i) for i in id2]
id2_label = [str(i) for i in label2]
id2_not_in_id1 = [i for i in id2_str if i not in id1_str]
cleaning = [i for i in zip(id2_str, id2_label) if str(i[0]) in id2_not_in_id1 and str(i[1]) != 'nan']
id2_cleaning = [i[0] for i in cleaning]
print(id2_cleaning)
label_cleaning = [i[1] for i in cleaning]
cleaning_dict = dict(cleaning)
#print(id2_cleaning)
sub_dirs = ['2022-61', '2022-81', '2022-84']
with open(part2_dataset, 'w') as fp:
    for sb in tqdm(sub_dirs):
        for sbf in os.listdir(os.path.join(part2_data_root, sb)):
            if str(sbf) in id2_cleaning:
                for idx, img in enumerate(os.listdir(os.path.join(part2_data_root, sb, sbf, 'Merged-1'))):
                    shutil.copy(os.path.join(part2_data_root, sb, str(sbf), 'Merged-1', img), os.path.join(dst, str(sbf) + '_' + str(idx) + '.tif'))
                    fp.writelines(sbf + '_' + str(idx) + '.tif' + ',' + str(int(float(cleaning_dict[sbf]))) + '\n')
    


