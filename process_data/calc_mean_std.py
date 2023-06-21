import os
import cv2
from tqdm import tqdm
import numpy as np

def cal_mean_std(images_dir):
    """
    给定数据图片根目录,计算图片整体均值与方差
    :param images_dir:
    :return:
    """
    img_filenames = [i for i in os.listdir(images_dir) if i !='.DS_Store']
    m_list, s_list = [], []
    for img_filename in tqdm(img_filenames):
        img = cv2.imread(images_dir + '/' + img_filename)
        img = img / 255.0
        m, s = cv2.meanStdDev(img)

        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
        #print(m_list)
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print('mean: ',m[0][::-1])
    print('std:  ',s[0][::-1])
    return m

if __name__=='__main__':
    cal_mean_std('/home/wangyin/data/mIHC')
