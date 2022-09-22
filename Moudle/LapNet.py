import time
import sys
import os
sys.path.append('../')
import torch.nn as nn
import torch.nn.functional as F
import torch
from Moudle.unet_model import UNet
from Moudle.carefe import CARAFE
# def default_conv(in_channels, out_channels, kernel_size, bias=True):
#     return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

"""
w/out  upsample functional kernel
"""


class CALayer(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction=4):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size ))
        modules_body.append(nn.PReLU())
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.CA = CALayer(n_feat, reduction)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, channels=3):  # 高斯核
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.cuda()
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):  # 高斯卷积
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):  # 拉普拉斯金字塔高频分量和低频分量的区分
        current = img
        pyr = []
        pyr_ori = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up # 高频残差
            # print(diff.shape)
            # print(current.shape)
            # print('*****')
            pyr.append(diff)
            pyr_ori.append(current)
            current = down
        pyr.append(current)
        pyr_ori.append(current)
        #print(pyr[-2].shape,pyr_ori[-1].shape) #torch.Size([1, 3, 128, 128]) torch.Size([1, 3, 128, 128])  pyr[-1]=64
        return pyr,pyr_ori

    def pyramid_recons(self, pyr):  # list
        image = pyr[-1]
        pyr_list = []
        for level in reversed(pyr[:-1]):  # 反向遍历除了最后一个元素
            up = nn.functional.interpolate(image,mode='bilinear',scale_factor=2)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = level
            # print(image.shape)
            pyr_list.append(image)
        return pyr_list
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias=False):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)
        
        self.iluec = conv(n_feat,3, kernel_size)
        self.iludc = conv(3,n_feat, kernel_size)
        
        self.recomnet = ORB(n_feat,in_features=6 ,kernel_size=kernel_size)
    def forward(self, x, x_img):        #x :feature x_img:original img
        ilumap = self.iluec(x)
        ilufeat = self.iludc(ilumap)
        x1 = x
        lap_map = self.conv2(x)
        img =  self.recomnet(torch.cat((lap_map,x_img),dim=1))

        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+ilufeat
        return x1, img,ilumap,lap_map

### RecomMoudle
class RecomMoudle(nn.Module):
    def __init__(self,n_feat, kernel_size,bias=False):
        super(RecomMoudle, self).__init__()
        self.unet = UNet(n_feat,3)
        self.iluec = conv(n_feat,3, kernel_size)   
        self.recomnet = ORB(72,in_features=6, kernel_size=kernel_size)
    def forward(self,x, x_img):
        ilumap = self.iluec(x)
        lap_map = self.unet(x)
        img = self.recomnet(torch.cat((lap_map,x_img),dim=1))
        return img,ilumap,lap_map
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, in_features=3,kernel_size=3, reduction=4,num_cab=8):
        super(ORB, self).__init__()
        modules_body = []
        modules_body += [conv(in_features,n_feat, kernel_size)]
        modules_body += [CAB(n_feat, kernel_size, reduction) for _ in range(num_cab)]
        modules_body.append(nn.PReLU())
        modules_body.append(conv(n_feat, 3, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        # res += x
        return res

class Trans_high(nn.Module):
    def __init__(self,  num_high=4):
        super(Trans_high, self).__init__()
        # self.fextact0 = nn.Sequential(conv(9,9,3),CAB(9,3))
        self.fextact1 = nn.Sequential(conv(3,18,3),CAB(18,3))
        self.fextact2 = nn.Sequential(conv(3,36,3),CAB(36,3))
        self.sam1 = SAM(18, 3)
        self.sam2 = SAM(36, 3)
        self.sam3 = SAM(72,3)
        #  enhance original
        self.fextact3 = nn.Sequential(conv(3,64,3),CAB(64,3))
        self.conv1 = conv(64,3,3)
        self.carefe1 = CARAFE(18)
        self.carefe2 = CARAFE(36)
        self.num_high = num_high
        #phase1
        self.model = UNet(9,18)
        #phase2
        self.trans_mask_block2 = UNet(36,36)
        #phase3
        # self.trans_mask_block3 = UNet(72,3)
        #ori enhance model
        self.orirecom = RecomMoudle(72,3)
    def forward(self, x, pyr_lap, fake_low,pyr_high):   # concat分量 lp分量list 低频处理分量
        pyr_result = []
        pyr_lap1 = []
        pyr_ilu = []
        # mask = (self.fextact0(x))
        mask = self.model(x)  # 算一个掩码出来
        # print(mask.shape)
        feature2,result_highfreq2,ilumap2,lap2 = self.sam1(mask,pyr_high[-2])
        setattr(self, 'result_highfreq_{}'.format(str(0)), result_highfreq2) #torch.Size([1, 3, 64, 64])
        # print(self.result_highfreq_0.shape)
        # feature2 = nn.functional.interpolate(feature2,size=(pyr_lap[-3].shape[2], pyr_lap[-3].shape[3]))
        feature2 = self.carefe1(feature2)
        if feature2.shape[2] != pyr_lap[-3].shape[2] or feature2.shape[3] != pyr_lap[-3].shape[3]:
            feature2 = nn.functional.interpolate(feature2, size=(pyr_lap[-3].shape[2], pyr_lap[-3].shape[3]))
        feature3 = torch.cat((feature2,self.fextact1(pyr_lap[-3])),dim=1)
        feature3 = self.trans_mask_block2(feature3)
        temp0 = feature3
        if feature3.shape[2] != pyr_high[-3].shape[2] or feature3.shape[3] != pyr_high[-3].shape[3]:
            feature3 = nn.functional.interpolate(feature3, size=(pyr_high[-3].shape[2], pyr_high[-3].shape[3]))
        feature3,result_highfreq3,ilumap3,lap3 = self.sam2(feature3,pyr_high[-3])
        temp = feature3
        setattr(self, 'result_highfreq_{}'.format(str(1)), result_highfreq3) #torch.Size([1, 3, 128, 128])
        # feature3 = nn.functional.interpolate(feature3, size=(pyr_lap[-4].shape[2], pyr_lap[-4].shape[3]))
        feature4 = self.carefe2(feature3)
        if feature4.shape[2] != pyr_high[-4].shape[2] or feature4.shape[3] != pyr_high[-4].shape[3]:
            feature4 = nn.functional.interpolate(feature4, size=(pyr_high[-4].shape[2], pyr_high[-4].shape[3]))
        feature4 = (torch.cat((feature4,self.fextact2(pyr_lap[-4])),dim=1))
        # result_highfreq4 = self.conv(self.orb(feature4))
        # feature4 = self.trans_mask_block3(feature4)
        # result_highfreq4 = feature4 + pyr_high[-4]
        # feature5,result_highfreq4,lap4 = self.sam3(feature4,pyr_high[-4])
        # orifeature = self.fextact3(pyr_high[-4])
        # fuse = orifeature+feature5
        # fuse2 = self.orirecom(fuse)
        result_highfreq4,ilumap4,lap4 = self.orirecom(feature4,pyr_high[-4])
        setattr(self, 'result_highfreq_{}'.format(str(2)), result_highfreq4)  # torch.Size([1, 3, 256, 256])
        pyr_lap1.append(lap2)
        pyr_lap1.append(lap3)
        pyr_lap1.append(lap4)
        pyr_ilu.append(ilumap2)
        pyr_ilu.append(ilumap3)
        pyr_ilu.append(ilumap4)

        for i in reversed(range(self.num_high)):
            result_highfreq = getattr(self, 'result_highfreq_{}'.format(str(i)))
            pyr_result.append(result_highfreq)

        pyr_result.append(fake_low)  # 低频分量追加到后面即可
        #print((pyr_result[0].shape))
        return pyr_result,pyr_lap1,pyr_ilu,temp,temp0

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class LapNet(nn.Module):
    def __init__(self, num_high=3):
        super(LapNet, self).__init__()
        self.lap_pyramid = Lap_Pyramid_Conv(num_high)
        trans_high = Trans_high(num_high=num_high)
        self.trans_high = trans_high
        self.unet = UNet(3,3)
        self.carefe0 = CARAFE(3)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
       
    def forward(self, real_A_full):
        pyr_A,pyr_O = self.lap_pyramid.pyramid_decom(img=real_A_full) #pyr_a is lap,pyr_0 is ori
        # print((pyr_O[0].shape))
        # print((pyr_A[0].shape)) #pyr_A[0] 1,3,512,512
        fake_B_low = self.unet(pyr_A[-1])   #last nomal map

        real_A_up = self.carefe0(pyr_A[-1])
        if real_A_up.shape[2] != pyr_A[-2].shape[2] or real_A_up.shape[3] != pyr_A[-2].shape[3]:
            real_A_up = nn.functional.interpolate(real_A_up, size=(pyr_A[-2].shape[2], pyr_A[-2].shape[3]))
        # fake_B_up = nn.functional.interpolate(fake_B_low, size=(pyr_A[-2].shape[2], pyr_A[-2].shape[3]))
        fake_B_up = self.carefe0(fake_B_low)
        if fake_B_up.shape[2] != pyr_A[-2].shape[2] or fake_B_up.shape[3] != pyr_A[-2].shape[3]:
            fake_B_up = nn.functional.interpolate(fake_B_up, size=(pyr_A[-2].shape[2], pyr_A[-2].shape[3]))
        high_with_low = torch.cat([pyr_A[-2], real_A_up, fake_B_up], 1)
        #print(high_with_low.shape)
        pyr_A_trans,pyr_lap1,ilumap,temp,temp0= self.trans_high(high_with_low, pyr_A, fake_B_low,pyr_O)  # list concat分量 lp分量list 低频处理分量
        # print(temp0.shape)
        pyr_result = self.lap_pyramid.pyramid_recons(pyr_A_trans)
        # pyr_result = [self.sig(item) for item in pyr_result]
        pyr_result.insert(0,self.sig(fake_B_low))  #[64,128,256,512]
        # output = self.orb(pyr_O[0],pyr_result)
        fake_B_full = pyr_result[-1]
        # print(fake_B_full.shape)
        return fake_B_full,pyr_result,ilumap,pyr_lap1



if __name__ == "__main__":
    device = 'cuda'
    X = torch.Tensor(1,3,512,512).cuda()
    net = LapNet(num_high=3).cuda()
    # net = nn.DataParallel(net)
    y = net(X)
    lap_list = y[-1]
    for img in lap_list:
        print(img.shape)
