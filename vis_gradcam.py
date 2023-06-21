from torchcam.methods import SmoothGradCAMpp
from dataset import mIHC_image_path
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from utils import get_network, get_test_dataloader
from torchcam.utils import overlay_mask
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-mode', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-test-file', type=str, default='./data/test_part1.txt', help='batch size for dataloader')
    parser.add_argument('-train-file', type=str, default='./data/train_part1.txt', help='batch size for dataloader')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-plot-auc', type=bool, default=True, help='plot auc image')
    parser.add_argument('-gen-output', type=bool, default=True, help='generate output')
    args = parser.parse_args()
    TRAIN_MEAN, TRAIN_STD = [0.12206179, 0.09673566, 0.20968879], [0.12873057, 0.13296286, 0.2005473]

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
    ])
    test_datasets = mIHC_image_path(file_path=args.test_file, transform=transform_train)
    test_loader = DataLoader(test_datasets, shuffle=False, num_workers=1, batch_size=1)
    train_datasets = mIHC_image_path(file_path=args.train_file, transform=transform_train)
    train_loader = DataLoader(train_datasets, shuffle=False, num_workers=1, batch_size=1)

    if args.mode=='tst':
        data_loader = test_loader
        save_dir = 'SmoothGradcamVisTest'
    if args.mode=='train':
        data_loader = train_loader
        save_dir = 'SmoothGradcamVisTrain'
    #org_image, image_rs, image, path = test[0]

    model = get_network(args)
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    cam_extractor = SmoothGradCAMpp(model)
    for n_iter, (org_image, image_rs, image, path) in tqdm(enumerate(data_loader)):
        _, c, h, w = image.shape
        patchs = []
        img_nm = path[0].split('/')[-1].split('.')[0]
        '''
        if args.gpu:
            image = image.cuda()
            #label = label.cuda()
            #print('GPU INFO.....')
            #print(torch.cuda.memory_summary(), end='')
        #print(image.size())
        out = model(image)
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
        print(activation_map[0].shape)
        result = overlay_mask(to_pil_image(image_rs.squeeze(0).numpy()), to_pil_image(activation_map[0], mode='F'), alpha=0.5)
        plt.imshow(activation_map[0].squeeze(0).detach().cpu().numpy()); plt.axis('off'); plt.tight_layout();
        plt.imshow(result); plt.axis('off'); plt.tight_layout(); 
        plt.savefig('./{}.png'.format(img_nm))
        '''
        for i in range(4):
            for j in range(4):
                #torch.cuda.empty_cache()
                image_patch = image[:, :, int(256*i):int(256*(i+1)), int(348*j):int(348*(j+1))]
                #print(image_patch.shape)
                out = model(image_patch.cuda())
                #print(out.size())
                out = out.detach().cpu().numpy()
                activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
                #print(activation_map[0].shape)
                patchs.append(activation_map[0].detach().cpu().numpy())
        #patchs = np.array(patchs).reshape((128, 176))
        #print(patchs.shape)
        patchs = np.concatenate([np.concatenate(patchs[i*4:(i+1)*4], axis=1) for i in range(4)])
        #print(patchs.shape)
        #print(activation_map[0].size())
        # Resize the CAM and overlay it
        #print(image_rs.squeeze(0).shape)
        #result = overlay_mask(to_pil_image(image_rs.squeeze(0).numpy()), to_pil_image(activation_map[0], mode='F'), alpha=0.5)
        # Display it
        #plt.imshow(patchs); plt.axis('off'); plt.tight_layout(); 
        #plt.imshow(result); plt.axis('off'); plt.tight_layout(); 
        #plt.imshow(image_rs.squeeze(0)); plt.axis('off'); plt.tight_layout(); 
        #plt.savefig('./SmoothGradcamVis/{}.png'.format(img_nm))
        #plt.savefig('./{}.png'.format(img_nm))
        #np.save('{}/{}.npy'.format(save_dir, img_nm), activation_map[0].squeeze(0).detach().cpu().numpy())
        np.save('{}/{}.npy'.format(save_dir, img_nm), patchs)
