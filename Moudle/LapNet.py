import time
import sys
sys.path.append('../')
import torch.nn as nn
import torch.nn.functional as F
import torch
from Moudle.unet_model import UNet
# def default_conv(in_channels, out_channels, kernel_size, bias=True):
#     return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
"""
w/out  upsample functional kernel
"""
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
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
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
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
# class Lap_Pyramid_Bicubic(nn.Module):
#     """
#
#     """
#     def __init__(self, num_high=3):
#         super(Lap_Pyramid_Bicubic, self).__init__()
#
#         self.interpolate_mode = 'bicubic'
#         self.num_high = num_high
#
#
#     def pyramid_decom(self, img):
#         current = img
#         pyr = []
#         ori_pry = []
#         for i in range(self.num_high):
#             down = nn.functional.interpolate(current, size=(current.shape[2] // 2, current.shape[3] // 2), mode=self.interpolate_mode, align_corners=True)
#             ori_pry.append(down)
#             up = nn.functional.interpolate(down, size=(current.shape[2], current.shape[3]), mode=self.interpolate_mode, align_corners=True)
#             diff = current - up
#             pyr.append(diff)
#             current = down
#         pyr.append(current)
#         return pyr,ori_pry
#
#     def pyramid_recons(self, pyr):
#         image = pyr[-1]
#         for level in reversed(pyr[:-1]):
#             image = F.interpolate(image, size=(level.shape[2], level.shape[3]), mode=self.interpolate_mode, align_corners=True) + level
#         return image

class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda'), channels=3):  # 高斯核
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
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
        # img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        # out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return img

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

    def forward(self, x, x_img):        #x :feature x_img:original img
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img


class Trans_high(nn.Module):
    def __init__(self,  num_high=4):
        super(Trans_high, self).__init__()
        self.sam1 = SAM(9, 3)
        self.sam2 = SAM(18,3)
        self.num_high = num_high
        self.conv = nn.Conv2d(15, 3, kernel_size=1, ).cuda()
        #phase1
        model = [nn.Conv2d(9, 64, 3, padding=1)]
        model += [UNet(64,64)]
        model += [nn.Conv2d(64, 9, 3, padding=1)]
        self.model = nn.Sequential(*model)

        #phase2
        self.trans_mask_block2 = nn.Sequential(*model)
        self.trans_mask_block2[0] = nn.Conv2d(18,64,3,padding=1)
        self.trans_mask_block2[-1] = nn.Conv2d(64,18,3,padding=1)
        #phase3
        self.trans_mask_block3 = nn.Sequential(*model)
        self.trans_mask_block3[0] = nn.Conv2d(36, 64, 3, padding=1)
        self.trans_mask_block3[-1] = nn.Conv2d(64, 3,3,padding=1)

        #duplicate feature map

    def forward(self, x, pyr_lap, fake_low,pyr_high):   # concat分量 lp分量list 低频处理分量
        pyr_result = []
        mask = self.model(x)  # 算一个掩码出来
        #print(mask.shape)
            #self.trans_mask_block = getattr(self, 'trans_mask_block_{}'.format(str(i)))
        # mask = nn.functional.interpolate(mask, size=(pyr_lap[-2-i].shape[2], pyr_lap[-2-i].shape[3]))
            # result_highfreq = torch.mul(pyr_lap[-2-i], mask) + pyr_lap[-2-i]
            # result_highfreq = torch.mul(pyr_lap[-2-i], mask) + pyr_lap[-2-i]
        feature2,result_highfreq2 = self.sam1(mask,pyr_high[-2])
        setattr(self, 'result_highfreq_{}'.format(str(0)), result_highfreq2) #torch.Size([1, 3, 64, 64])
        # print(self.result_highfreq_0.shape)
        feature2 = nn.functional.interpolate(feature2,size=(pyr_lap[-3].shape[2], pyr_lap[-3].shape[3]))
        copyl1 = torch.cat((pyr_lap[-3],pyr_lap[-3],pyr_lap[-3]),dim=1)
        feature3 = torch.cat((feature2,copyl1),dim=1)
        feature3 = self.trans_mask_block2(feature3)
        feature3,result_highfreq3 = self.sam2(feature3,pyr_high[-3])
        setattr(self, 'result_highfreq_{}'.format(str(1)), result_highfreq3) #torch.Size([1, 3, 128, 128])
        feature3 = nn.functional.interpolate(feature3, size=(pyr_lap[-4].shape[2], pyr_lap[-4].shape[3]))
        copyl0 = torch.cat((pyr_lap[-4],pyr_lap[-4],pyr_lap[-4]),dim=1)
        copy = torch.cat((pyr_high[-4],pyr_high[-4],pyr_high[-4]),dim=1)
        feature4 = torch.cat((feature3,copyl0,copy),dim=1)
        # result_highfreq4 = self.conv(self.orb(feature4))
        result_highfreq4 = self.trans_mask_block3(feature4)
        result_highfreq4 = result_highfreq4 + pyr_high[-4]
        # feature4,result_highfreq4 = self.sam(feature4,pyr_high[-4])
        setattr(self, 'result_highfreq_{}'.format(str(2)), result_highfreq4)  # torch.Size([1, 3, 256, 256])


        for i in reversed(range(self.num_high)):
            result_highfreq = getattr(self, 'result_highfreq_{}'.format(str(i)))
            pyr_result.append(result_highfreq)

        pyr_result.append(fake_low)  # 低频分量追加到后面即可
        #print((pyr_result[0].shape))
        return pyr_result

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class LapNet(nn.Module):
    def __init__(self, num_high=3):
        super(LapNet, self).__init__()
        self.lap_pyramid = Lap_Pyramid_Conv(num_high)
        trans_high = Trans_high(num_high=num_high)
        self.trans_high = trans_high.cuda()
        self.unet = UNet(3,3).cuda()
        self.sam = SAM(3,3).cuda()
        self.sig = nn.Sigmoid().cuda()
        self.relu = nn.ReLU().cuda()
        # self.orb = ORB(64, 3).cuda()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1).cuda()
        self.conv2 = nn.Conv2d(64,3,kernel_size=3,stride=1,padding=1).cuda()
    def forward(self, real_A_full):

        pyr_A,pyr_O = self.lap_pyramid.pyramid_decom(img=real_A_full) #pyr_a is lap,pyr_0 is ori
        # print((pyr_O[-1].shape))
        # print((pyr_A[0].shape)) #pyr_A[0] 1,3,512,512
        fake_B_low = self.unet(pyr_A[-1])   #last nomal map
        # fake_B_low = self.relu((fake_B_low*pyr_A[-1])-fake_B_low+1)
        real_A_up = nn.functional.interpolate(pyr_A[-1], size=(pyr_A[-2].shape[2], pyr_A[-2].shape[3]))
        fake_B_up = nn.functional.interpolate(fake_B_low, size=(pyr_A[-2].shape[2], pyr_A[-2].shape[3]))
        high_with_low = torch.cat([pyr_A[-2], real_A_up, fake_B_up], 1)
        #print(high_with_low.shape)
        pyr_A_trans = self.trans_high(high_with_low, pyr_A, fake_B_low,pyr_O)  # list concat分量 lp分量list 低频处理分量
        pyr_result = self.lap_pyramid.pyramid_recons(pyr_A_trans)
        pyr_result = [self.sig(item) for item in pyr_result]
        pyr_result.insert(0,self.sig(fake_B_low))
        fake_B_full = pyr_result[-1]
        # print(fake_B_full.shape)
        return fake_B_full,pyr_result,pyr_A,fake_B_low



if __name__ == "__main__":
    device = torch.device("cuda")
    X = torch.Tensor(1,3,512,512).to(device)
    net = LapNet(num_high=3)
    #net = ORB(20,3).cuda()
    y = net(X)
