import argparse
import random
import torch
import os
import torchvision.transforms as tfs
from torch.testing._internal.distributed.rpc.examples.parameter_server_test import batch_size
from torchvision.transforms import functional as FF
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
import sys
sys.path.append('/home/xin/Experience/Dataload/')
import LoL
from LoL import TheDataset

#parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description='ConMixer')
parser.add_argument('--trainset',type=str,default='its_train')
parser.add_argument('--testset',type=str,default='its_test')
parser.add_argument('--qkv_bias', type=bool, default=True)
parser.add_argument('--perloss',default=True,action='store_true',help='perceptual loss')
parser.add_argument('--net',type=str,default='ConMixer')
#parser.add_argument('--model_dir',type=str,default= r'C:\jgl\yolov4\trained_moudles')
parser.add_argument('--device',type=str,default='cuda')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--epoch', default=90000, type=int, help='number of total epochs to run')
parser.add_argument('--warmup_epoch', default=10, type=int, help='the num of warmup epochs')
parser.add_argument('--init_lr', default=2e-4, type=float, help='a low initial learning rata for adamw optimizer')
parser.add_argument('--wd', default=0.5, type=float, help='a high weight decay setting for adamw optimizer')
parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'SGD'],
                    help='optimizer supported by PyTorch')
parser.add_argument('--save_path', default='weights', type=str, help='the path to saving the checkpoints')
parser.add_argument('--save_best', default=True, type=bool, help='saveing the checkpoint has the best acc')
parser.add_argument('--mixup', default=0.8, type=float, help='using mixup and set alpha value')
parser.add_argument('--crop_size', type=int, default=256, help='Takes effect when using --crop ')
parser.add_argument('--no_lr_sche',default=False,action='store_true',help='no lr cos schedule')
opt = parser.parse_args()
# opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = opt.batch_size
crop_size = opt.crop_size

#
# class RESIDE_Dataset(data.Dataset):
#     def __init__(self, path, train, size=crop_size, format='.jpg'):
#         super(RESIDE_Dataset, self).__init__()
#         self.size = size
#         self.train = train                                                 # /imgs/haze/xxx.jpg
#         self.format = format                                               # haze下面一堆文件夹,每个文件夹下面有两个文件夹 一个是haze 一个是clear
#         self.haze_imgs_dir = os.listdir(os.path.join(path, 'haze'))
#         self.haze_imgs = [os.path.join(path, 'haze', img) for img in self.haze_imgs_dir]
#         self.clear_dir = os.path.join(path, 'clear')
#
#     def __getitem__(self, index):
#         haze = Image.open(self.haze_imgs[index])
#         if isinstance(self.size, int):
#             while haze.size[0] < self.size or haze.size[1] < self.size:     # 如果长和高小于裁剪尺寸，换一张打开它 直到他的高和宽都比裁剪尺寸大
#                 index = random.randint(0, 20000)
#                 haze = Image.open(self.haze_imgs[index])
#         img = self.haze_imgs[index]                             # img就是这张haze的图片
#     #    id = img.split('\\')[-1].split('_')[0]                  # id是格式化以后的序号
#         id = img.split('\\')[-1]                # id是格式化以后的序号
#         # print('id'+id)
#     #    clear_name = id + self.format                           # 没有雾的图片名就是上面这个序号+.png  这个和有雾图片是一一对应的
#         clear_name = id
#         # print('clearName:'+self.clear_dir+'\\'+clear_name)
#         clear = Image.open(os.path.join(self.clear_dir, clear_name))    # 打开有雾图片对应的没有雾的图片
#         clear = tfs.CenterCrop(haze.size[::-1])(clear)          # 按照有雾图像的相反尺寸裁剪无雾图像
#         if not isinstance(self.size, str):                      # 如果传进来的size不是字符串，
#             i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size[0], self.size[1]))  # 在这张有雾图片的任意位置按照给定尺寸剪裁
#             haze = FF.crop(haze, i, j, h, w)                    # 按照这个任意尺寸裁剪一张有雾图片
#             clear = FF.crop(clear, i, j, h, w)                  # 按照这个任意尺寸裁剪一张无雾图片
#         haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))   # 把这两张图片转换成RGB图像返回
#         return haze, clear
#
#     def augData(self, data, target):
#         if self.train:
#             rand_hor = random.randint(0, 1)
#             rand_rot = random.randint(0, 3)
#             data = tfs.RandomHorizontalFlip(rand_hor)(data)
#             target = tfs.RandomHorizontalFlip(rand_hor)(target)
#             if rand_rot:
#                 pass
#                 # data = FF.rotate(data, 90 * rand_rot)
#                 # target = FF.rotate(target, 90 * rand_rot)
#         data = tfs.ToTensor()(data)
#         data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
#         target = tfs.ToTensor()(target)
#         return data, target
#
#     def __len__(self):
#         return len(self.haze_imgs)
#
#
# path = r'C:\Users\User\Desktop\4kdataset'
# ITS_train_loader = DataLoader(dataset=RESIDE_Dataset(path , train=True, size=[2160,3840]), batch_size=batch_size,shuffle=True)
# ITS_test_loader = DataLoader(dataset=RESIDE_Dataset(path + '/test/', train=False, size=[2160,3840]), batch_size=batch_size,shuffle=True)
route = '/home/xin/Experience/dataset/ADOBE5K/'
ITS_train_loader = DataLoader(dataset=TheDataset(route=route), batch_size=batch_size,shuffle=True)
ITS_test_loader = DataLoader(dataset=TheDataset(route=route,phase='test'), batch_size=batch_size,shuffle=True)
# ITS_eval_loader = DataLoader(dataset=TheDataset(route=route,phase='eval'), batch_size=batch_size,shuffle=True)

if __name__ == "__main__":
   pass
