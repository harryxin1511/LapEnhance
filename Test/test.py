import os,argparse
import numpy as np
import torchvision
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import sys
sys.path.append('../')
sys.path.append('../')
abs=os.getcwd()+'/'




#         for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
#             img = make_grid(tensor)
#             npimg = img.numpy()
#             ax = fig.add_subplot(221+i)
#             ax.imshow(np.transpose(npimg, (1, 2, 0)))
#             ax.set_title(tit)
#         plt.show()

parser=argparse.ArgumentParser()
parser.add_argument('--task',type=str,default='its',help='its or ots')
parser.add_argument('--test_imgs',type=str,default='test_imgs',help='Test imgs folder')
parser.add_argument('--decom',type=int,default=False)
parser.add_argument('--mask',type=int,default=False)

opt=parser.parse_args()
dataset=opt.task

img_dir='/home/xin/Experience/dataset/Adobe5K/test/low/'
# img_decom_dir='/home/xin/Experience/LapEnhace/Test/test_imgs/'
output_dir='../Test/20.89colloss/'
output_decom = '../Test/Decom/'
output_mask = '../Test/mask/'
print("pred_dir:",output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

device='cuda' if torch.cuda.is_available() else 'cpu'

net = torch.load('../trained_moudles/ll1999.pth')
print(type(net))

net.eval()
for im in os.listdir(img_dir):
    print(f'\r {im}',end='',flush=True)
    haze = Image.open(img_dir+im)
    haze1= tfs.Compose([
        tfs.ToTensor(),
        #tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
    ])(haze)[None,::]
    haze_no=tfs.ToTensor()(haze)[None,::]  # ?
    with torch.no_grad():
        haze1 = haze1.cuda()
        resize=torchvision.transforms.Resize(512)
        haze1=resize(haze1)
        pred= net(haze1)
        for idx,image in enumerate(pred[1]):
            tss = torch.squeeze(image.clamp(0,1).cpu())
            vutils.save_image(tss, output_mask + im.split('.')[0] + f'_Lap{idx}.png')
        # print(mask)
        if opt.decom:
            for idx,img in enumerate(temp):
                t = torch.squeeze(img.clamp(0, 1).cpu())
            # tensorShow([haze_no,pred.clamp(0,1).cpu()],['haze','pred'])
                vutils.save_image(t, output_decom + im.split('.')[0] + f'_Lapdecom{idx}.png')
        if opt.mask:
            for idx, img in enumerate(mask):
                t = torch.squeeze(img.clamp(0, 1).cpu())
                # tensorShow([haze_no,pred.clamp(0,1).cpu()],['haze','pred'])
                vutils.save_image(t, output_mask + im.split('.')[0] + f'_Lapmask{idx}.png')
    ts=torch.squeeze(pred[0].clamp(0,1).cpu())
    # lapdawn=torch.squeeze(pred[-1].clamp(0,1).cpu())
    #tensorShow([haze_no,pred.clamp(0,1).cpu()],['haze','pred'])
    vutils.save_image(ts,output_dir+im.split('.')[0]+'_Lap.png')
    # vutils.save_image(lapdawn,output_dir+im.split('.')[0]+'_Lapdown.png')

