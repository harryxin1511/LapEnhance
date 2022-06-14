import numpy as np
import time
import torch
import sys,os
sys.path.append('../')
sys.path.append('/home/xin/Experience/LapDehaze/data/')
from torch import nn, optim
from torchvision import transforms
from Moudle.carefe import CARAFE
from Moudle.LapNet import LapNet
from Moudle.LapNet import Lap_Pyramid_Conv
from data.lr_scheduler import lr_schedule_cosdecay
from data.metrics import ssim, psnr
from data.option import opt, ITS_train_loader, ITS_test_loader
import lib.pytorch_ssim as pytorch_ssim
from lib.utils import TVLoss, print_network
from data.losses import ColorLoss,Blur
save_test_path = './TestResult/'
save_ori_path = './Ori/'
device_id =[]
if not os.path.exists('/home/xin/Experience/drive/srmnet4.4'):
    os.mkdir('/home/xin/Experience/drive/srmnet4.4')
from torch.nn.modules.loss import  _Loss
from torchvision.models import vgg
import pandas as pd
import torch.nn.functional as F
# df = pd.DataFrame(columns=['step','ssim','psnr'])
# df.to_csv('./result.csv')
model = {
    'ConMixer':LapNet(num_high=3)
}
loaders = {
    'its_train': ITS_train_loader,
    'its_test': ITS_test_loader
}
from visdom import Visdom
viz = Visdom(env='feature_maps')
"""
损失函数设置为gt和sam的输出，不是与上采样的家和

"""
def train(loader_train,loader_test,net,optimizer):
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
    decomori = Lap_Pyramid_Conv()
    count =0
    print('train from scratch---------------------------')
    for epoch in range(opt.epoch):
        lr = opt.init_lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(epoch, opt.epoch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        x, y = next(iter(loader_train))
        x = x.to(opt.device)
        # print(x.shape)
        # print(x.shape)
        y = y.to(opt.device)
        net = net.to(opt.device)
        # net = torch.nn.DataParallel(net,de)
        start=time.time()
        out,pyr_list,pyr_lap,pyr_lape = net(x)
        end = time.time()
        #extact each pyr img
        Scale0 = pyr_list[3] #512
        Scale1 = pyr_list[2] #256
        Scale2 = pyr_list[1] #128
        Scale3 = pyr_list[0] #64
        #gt_down
        orilist = decomori.pyramid_decom(y)[1]
        gt_down1 = orilist[1]  # 256
        gt_down2 = orilist[2] # 128
        gt_down3 = orilist[3]  # 64

        reup1 = F.interpolate(gt_down3, scale_factor=2, mode='bilinear')  # 128
        reup2 = F.interpolate(gt_down2, scale_factor=2, mode='bilinear')  # 256
        reup3 = F.interpolate(gt_down1, scale_factor=2, mode='bilinear')  # 512
        #lap_gt
        lapgtlist = decomori.pyramid_decom(y)[0]
        lap0_gt = lapgtlist[-2]  # 128
        lap1_gt = lapgtlist[-3]
        lap2_gt = lapgtlist[-4]
        # print(lap1_gt.shape)
        # lap_enhance
        lap0 = pyr_lape[0]  # 128
        lap1 = pyr_lape[1]  # 256
        lap2 = pyr_lape[2]  # 512
        # print(lap1.shape)
        """lap loss"""
        laploss0 = L1_criterion(lap0_gt, lap0)
        laploss1 = L1_criterion(lap1_gt, lap1)
        laploss2 = L1_criterion(lap2_gt, lap2)
        total_laploss = laploss0 + laploss1 + laploss2
        #loss = criterion[0](out,y)
        """ l1 loss """
        scale0l1 = L1_closs(Scale0,y)  #512
        scale1l1 = L1_closs(Scale1,gt_down1) #256
        scale2l1 = L1_closs(Scale2,gt_down2) #128
        scale3l1 = L1_closs(Scale3,gt_down3) #64
        scaleloss = scale0l1 + scale1l1 + 2*scale2l1 + 2*scale3l1
        """color_loss """
        blur_rgb = Blur(3).cuda()
        inputc = blur_rgb(Scale0)
        labelc = blur_rgb(y)
        color_loss1 = color_loss(inputc,labelc)
        """ssim loss"""
        ssim_loss1 = 1 - ssim(out, y)
        ssim_loss2 = 1 - ssim(Scale1, gt_down1)
        ssim_loss3 = 1 - ssim(Scale2, gt_down2)  # 128
        ssim_loss4 = 1 - ssim(Scale3, gt_down3)  # 64
        ssim_loss = ssim_loss1 + ssim_loss2 +  2*ssim_loss3 + 2*ssim_loss4
        # ssim_loss = 1 - ssim(out, y)
        """tv_loss"""
        tv_loss = TV_loss(out)
        """vgg loss"""
        viz.images(lap2,win='pred')
        viz.images(lap2_gt,win='gt')

        loss = scaleloss +ssim_loss +total_laploss
        # loss = scaleloss
        #ssim_loss + 0.01 * tv_loss
        # AvgScale0Loss = AvgScale0Loss + torch.Tensor.item(scale0l1.data)
        # AvgScale1Loss = AvgScale1Loss + torch.Tensor.item(scale1l1.data)
        # AvgScale2Loss = AvgScale2Loss + torch.Tensor.item(scale2l1.data)
        # AvgScale3Loss = AvgScale3Loss + torch.Tensor.item(scale3l1.data)
        # AvgColor0Loss = AvgColor0Loss + torch.Tensor.item(scale0color.data)
        # AvgColor1Loss = AvgColor1Loss + torch.Tensor.item(scale1color.data)
        # AvgColor2Loss = AvgColor2Loss + torch.Tensor.item(scale2color.data)
        # AvgColor3Loss = AvgColor3Loss + torch.Tensor.item(scale3color.data)
        # count += 1
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # viz.line([loss.item()],[epoch],win='train loss',update='append')
        print( f'\rtrain loss : {loss.item():.5f}| step :{epoch}/{opt.epoch}|lr :{lr :.7f} |time_used :{(end - start)  :}',end='', flush=True)


        if (epoch+1) % 1000 == 0:
            print('\n ----------------------------------------test!-----------------------------------------------------------')
            with torch.no_grad():
                ssim_eval, psnr_eval = test(net, loader_test,epoch)
                print(f'\nstep :{epoch} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')
                ssims.append(ssim_eval)
                psnrs.append(psnr_eval)
                if ssim_eval > max_ssim and psnr_eval > max_psnr:
                    max_ssim = max(max_ssim, ssim_eval)
                    max_psnr = max(max_psnr, psnr_eval)
                # torch.save(net.state_dict(),opt.model_dir+'/train_model.pth')
                torch.save(net,f'/home/xin/Experience/drive/srmnet4.4/ll{epoch}.pth')
                list = [epoch, ssim_eval, psnr_eval ]
                data = pd.DataFrame([list])
                data.to_csv('./srm_result.csv',mode='a')
                # print(opt.model_dir+'/train_model.pth')
                print(f'\n model saved at step :{epoch}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')



def test(net,loader_test,epoch):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    #inputs, targets = next(iter(loader_test))
    # for i in range(100):
    for idx,data in enumerate(loader_test,1):
            inputs,targets =data[0],data[1]
            inputs = inputs.to(opt.device)
            targets = targets.to(opt.device)
            pred,pyrlist,pyr_A,pyr_Atrans = net(inputs)  # 预测值
            ssim1 = ssim(pred, targets).item()  # 真实值
            psnr1 = psnr(pred, targets)
            ssims.append(ssim1)
            psnrs.append(psnr1)
            # toPIL = transforms.ToPILImage()
            # img = pred[0]
            # # print(img.shape)
            # img1 = targets[0]
            # pic = toPIL(img)
            # pic1 = toPIL(img1)
            # pic.save(save_test_path+f'{epoch}'+f'pre{idx}.jpg')
            # pic1.save('clear.jpg')
    return np.mean(ssims), np.mean(psnrs)



class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class PerceptualLoss(_Loss):
    def __init__(self, ):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg.vgg19(pretrained=True).features
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.vgg.eval()

    def vgg_forward(self, x):
        output = []
        for name, module in self.vgg._modules.items():
            x = module(x)
            if name == '26':
                return x

    def preprocess(self, tensor):
        trsfrm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
        res = trsfrm(tensor)
        return res

    def forward(self, output, label):
        output = self.preprocess(output)
        label = self.preprocess(label)
        feat_a = self.vgg_forward(output)
        feat_b = self.vgg_forward(label)

        return F.l1_loss(feat_a, feat_b)


#TEST!
if __name__ == "__main__":

    torch.cuda.empty_cache()
    loader_train = loaders[opt.trainset]    # its_train
    loader_test = loaders[opt.testset]      # its_test
    net = model[opt.net]
    # if torch.cuda.device_count() > 0:
    #     net = torch.nn.DataParallel(net,device_ids=[0,1])
    net = net.to(opt.device)

    """
    LOSS
    """
    L1_criterion = nn.L1Loss()
    L1_closs = L1_Charbonnier_loss()
    # L1_closs = torch.nn.L1Loss()
    TV_loss = TVLoss()
    mse_loss = torch.nn.MSELoss()
    vgg_loss = PerceptualLoss()
    color_loss = ColorLoss()

    ssim = pytorch_ssim.SSIM()
    if torch.cuda.is_available():
        mse_loss = mse_loss.cuda()
        L1_criterion = L1_criterion.cuda()
        L1_closs = L1_closs.cuda()
        TV_loss = TV_loss.cuda()
        vgg_loss = vgg_loss.cuda()
        ssim = ssim.cuda()
        color_loss = color_loss.cuda()

    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.init_lr, betas=(0.9, 0.999),
                            eps=1e-08)
    optimizer.zero_grad()
    train(loader_train,loader_test,net,optimizer)