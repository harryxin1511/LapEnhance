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

sys.path.append('../')
from data.metrics import psnr,ssim
from Moudle.LapNet import LapNet
abs=os.getcwd()+'/'
parser=argparse.ArgumentParser()
parser.add_argument('--task',type=str,default='its',help='its or ots')
parser.add_argument('--feature',type=int,default=True,help='Test imgs folder')
parser.add_argument('--decomori',type=int,default=False)
parser.add_argument('--mask',type=int,default=False)

opt=parser.parse_args()
dataset=opt.task

# img_dir = '/home/xin/Experience/dataset/LOLdataset/test/low/'
# normal_dir = '/home/xin/Experience/dataset/LOLdataset/test/high/'
# img_dir = '/home/xin/Experience/dataset/fkc/test/low/'
img_dir = '/home/xin/Experience/high_resolution/input/'
# normal_dir = '/home/xin/Experience/dataset/fkc/test/high/'
normal_dir = '/home/xin/Experience/high_resolution/expertC_gt/'
output_dir='../Test/VVresult/'
output_decomori = '../Test/DecomLOW/'
output_mask = '../Test/illumap/'
output_features = '../Test/25.47featuremap/'
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
net.load_state_dict(torch.load('/home/xin/Experience/drive/srmfnet/ll59999.pth'))
# net = Lap_Pyramid_Conv()
net.eval()
psnrs = []
ssims = []
for im,om in zip(os.listdir(img_dir),os.listdir(normal_dir)):
    print(f'\r {im}',end='',flush=True)
    torch.cuda.empty_cache()
    total = []
    haze = Image.open(img_dir+im)
    haze_no = Image.open(normal_dir+om)
    haze1= tfs.Compose([
        tfs.ToTensor(),
        # tfs.Resize([512,512])
        #tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
    ])(haze)[None,::]
    haze_no1=tfs.Compose([
        tfs.ToTensor(),
        tfs.Resize([512,512])
        ])(haze_no)[None,::]  # ?

    with torch.no_grad():
        haze1 = haze1.cuda()
        haze_no1=haze_no1.cuda()
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
        ssim1 = ssim(pred[0], haze_no1).item()  # 真实值
        psnr1 = psnr(pred[0], haze_no1)
        ssims.append(ssim1)
        psnrs.append(psnr1)
    compare_img = torch.cat([haze1,pred[0],haze_no1],dim=3)
    # compare_img = torch.cat([haze1,pred[0]],dim=3)
    # compare_img = pred[0]
    ts=torch.squeeze(compare_img.clamp(0,1).cpu())
    # ts=torch.squeeze(pred[0].clamp(0,1).cpu())
    # lapdawn=torch.squeeze(pred[-1].clamp(0,1).cpu())
    #tensorShow([haze_no,pred.clamp(0,1).cpu()],['haze','pred'])
    vutils.save_image(ts,output_dir+im.split('.')[0]+'_Lap.png')
    # vutils.save_image(lapdawn,output_dir+im.split('.')[0]+'_Lapdown.png')
print(f'avg psnr:{np.mean(psnrs)},avg ssim:{np.mean(ssims)}')
