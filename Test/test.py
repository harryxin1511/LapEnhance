import os,argparse
import numpy as np
import torchvision
from PIL import Image
import torch
import math
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from piq import psnr
sys.path.append('../')
sys.path.append('../')
from Moudle.LapNet import LapNet
abs=os.getcwd()+'/'
parser=argparse.ArgumentParser()
parser.add_argument('--task',type=str,default='its',help='its or ots')
parser.add_argument('--feature',type=int,default=True,help='Test imgs folder')
parser.add_argument('--decomori',type=int,default=False)
parser.add_argument('--mask',type=int,default=True)

opt=parser.parse_args()
dataset=opt.task
# img_dir='/home/xin/Experience/dataset/单张/DICM/'
# dicm01 = '/home/xin/Experience/dataset/单张/DICM/01.jpg'
img_dir = '/home/xin/Experience/dataset/ADOBE5K/test/low/'
normal_dir = '/home/xin/Experience/dataset/ADOBE5K/test/high/'
# img_decom_dir='/home/xin/Experience/LapEnhace/Test/test_imgs/'
output_dir='../Test/R23.90v4.3/'
output_decomori = '../Test/DecomLOW/'
output_mask = '../Test/illumap/'
output_features = '../Test/23.90featuremap/'
print("pred_dir:",output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(output_mask):
    os.mkdir(output_mask)
if not os.path.exists(output_features):
    os.mkdir(output_features)

device='cuda'
net = LapNet().cuda()
# net = nn.DataParallel(net)
net.load_state_dict(torch.load('/home/xin/Experience/drive/srmfnet/ll185999.pth'))
# net = Lap_Pyramid_Conv()
net.eval()

for im in os.listdir(img_dir):
    print(f'\r {im}',end='',flush=True)

    total = []
    haze = Image.open(img_dir+im)
    # haze_no = Image.open(normal_dir+om)
    haze1= tfs.Compose([
        tfs.ToTensor(),
        #tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
    ])(haze)[None,::]
    # haze_no=tfs.ToTensor()(haze)[None,::]  # ?

    with torch.no_grad():
        haze1 = haze1.cuda()
        # haze_no=haze_no.cuda()
        # resize=torchvision.transforms.Resize(512)
        # haze1=resize(haze1)
        # pred= net.pyramid_decom(haze1) 
        pred= net(haze1)
        if opt.mask:
            for idx, img in enumerate(pred[-2]):
                t = torch.squeeze(img.clamp(0, 1).cpu())
                vutils.save_image(t, output_mask + im.split('.')[0] + f'_Lapmask{idx}.png')
        if opt.decomori:
            for idx, img in enumerate(pred[0]):
                t = torch.squeeze(img.clamp(0, 1).cpu())
                vutils.save_image(t, output_decomori + im.split('.')[0] + f'_Lapde{idx}.png')
        if opt.feature:
            img1 = pred[-1]
            for idx,img in enumerate(img1):
                #t = torch.squeeze(img.clamp(0, 1).cpu())
                img = img.clamp(0,1).squeeze(0).cpu()
                vutils.save_image(img, output_features + im.split('.')[0] + f'_Lapfe{idx}.png')

    ts=torch.squeeze(pred[0].clamp(0,1).cpu())
    # lapdawn=torch.squeeze(pred[-1].clamp(0,1).cpu())
    #tensorShow([haze_no,pred.clamp(0,1).cpu()],['haze','pred'])
    vutils.save_image(ts,output_dir+im.split('.')[0]+'_Lap.png')
    # vutils.save_image(lapdawn,output_dir+im.split('.')[0]+'_Lapdown.png')
    
