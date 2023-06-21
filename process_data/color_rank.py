import numpy as np
import os
from glob import glob
import json
import cv2
from scipy import stats
import pandas as pd

def contours_in(contours, canvas_size):
    p = np.zeros(shape=canvas_size)
    cv2.drawContours(p, contours, -1, 255, -1)
    a = np.where(p==255)[0].reshape(-1,1)
    b = np.where(p==255)[1].reshape(-1,1)
    coordinate = np.concatenate([a,b], axis=1).tolist()
    inside = [tuple(x) for x in coordinate]
    return inside

tif_files = '/home/wangyin/data/mIHC'
contours_files = glob('/home/wangyin/data/output/*/*/Merged-1/*')
test_npy_files = glob('../SmoothGradcamVisTrain/*.npy') + glob('../SmoothGradcamVisTest/*.npy')
partMap = '../data/mHICpart1Map.txt'
dst = './color_rank_total.csv'
#print(len(test_npy_files))
partMapFile = open(partMap, 'r')
mapLines = partMapFile.readlines()
mapDict = {}
for mapLine in mapLines:
    img_name, map_name = mapLine.strip().rsplit(',', 1)
    img_head = img_name.rsplit('.', 1)[0]
    for cf in contours_files:
        if img_head in cf:
            img_head = cf
            break
    map_head = map_name.rsplit('.', 1)[0]
    mapDict[map_head] = img_head

color_dict = {'yellow':[], 'orange':[], 'green':[], 'qing':[], 'blue':[], 'white':[], 'red':[]}
for test_npy_file in test_npy_files:
    npy_name = test_npy_file.split('/')[-1].split('.')[0]
    test_npy = np.load(test_npy_file)
    npy_h, npy_w = test_npy.shape
    json_name = mapDict[npy_name]
    org_img = cv2.imread(os.path.join(tif_files, npy_name + '.tif'))
    org_h, org_w, _ = org_img.shape
    #print(org_h, org_w)
    with open(json_name, 'r') as load_f:
        load_dict = json.load(load_f)
        for color in load_dict.keys():
            #print(color, len(load_dict[color].keys()))
            #print(len(list(load_dict[color].values())))
            #print(type(load_dict[color].values()))
            #cv2.drawContours(org_img, [np.array(i) for i in list(load_dict[color].values())], -1, (0, 255, 0), 3)
            ratio = npy_h / np.float64(org_h)
            contours = [np.asarray(np.array(i)*ratio, dtype=np.int32) for i in list(load_dict[color].values())]
            #print(len(contours))
            #n = 0
            #for i in range(len(contours)):
            #    print(len(contours[i]))
            #    n += len(contours[i])
            #print(n)
            inside = contours_in(contours, (npy_h, npy_w))
            for isd in inside:
                #print(isd)
                heat_value = test_npy[isd[0], isd[1]]
                color_dict[color].append(heat_value)
            #print(heat_value_list)
            #print(len(heat_value_list))
            #break
        #cv2.imwrite('./contours_img.tif', org_img)
    #break
data = {}
for k,v in color_dict.items():
    v = sorted(v)
    pc = int(0.05 * len(v))
    if pc>0:
        value = v[pc:-pc]
    else:
        value = v
    print(str(k) + '_mean: {}'.format(np.mean(value)))
    data[k] = np.mean(value)
df = pd.DataFrame.from_dict(data, orient='index', columns=['mean_heat_value'])
df.to_csv(dst)
