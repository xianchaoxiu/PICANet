import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
import os
import os.path as osp
import time
import numpy as np

from models import get_model


net = get_model('pica_resnet_fpn')
net.eval()
file_path =  "./datasets/sirst_aug/test/images"
pkl_file = r'./test_result/SIRST-Aug_FPN_mIoU_74.42.pkl'
checkpoint = torch.load(pkl_file, map_location=torch.device('cuda:1'))
net.load_state_dict(checkpoint)
net.eval()

imgDir = "./test_result/SIRST-Aug_FPN/img/"
if not os.path.exists(imgDir):
    os.makedirs(imgDir)
matDir = "./test_result/SIRST-Aug_FPN/mat/"
if not os.path.exists(matDir):
    os.makedirs(matDir)

for filename in os.listdir(file_path):
    
    img_gray = cv2.imread(file_path + '/' + filename, 0)
    img_gray = cv2.resize(img_gray, [256, 256],interpolation=cv2.INTER_LINEAR)
    img = img_gray.reshape(1, 1, 256, 256) / 255.
    img = torch.from_numpy(img).type(torch.FloatTensor)
    name = os.path.splitext(filename)[0]
    matname = name+'.mat'

    with torch.no_grad():
        net.mode = 'test'
        start = time.time()
        T = net(img)
        end = time.time()
        total = end - start
        T = T.detach().numpy().squeeze()
        T = (T > 0.5).astype(np.uint8) * 255
        


    cv2.imwrite(imgDir + filename, T)
    scio.savemat(matDir + matname, {'T': T})
