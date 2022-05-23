import numpy as np
import time
import torch
import sys,os
sys.path.append('../')
sys.path.append('/home/xin/Experience/LapDehaze/data/')
from torch import nn, optim
from torchvision import transforms
from Moudle.LapNet import LapNet
from data.lr_scheduler import lr_schedule_cosdecay
from data.metrics import ssim, psnr
from data.option import opt, ITS_train_loader, ITS_test_loader
import lib.pytorch_ssim as pytorch_ssim
from lib.utils import TVLoss, print_network
save_test_path = './TestResult/'
save_ori_path = './Ori/'
import pandas as pd
# df = pd.DataFrame(columns=['step','ssim','psnr'])
# df.to_csv('./result.csv')
model = {
    'ConMixer':LapNet(num_high=4)
}
loaders = {
    'its_train': ITS_train_loader,
    'its_test': ITS_test_loader
}


def train(loader_train,loader_test,net,optimizer):
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
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
        out = net(x)
        end = time.time()
        #loss = criterion[0](out,y)
        """
        loss
        """

        ssim_loss = 1 - ssim(out, y)
        tv_loss = TV_loss(out)
        loss = ssim_loss + 0.01 * tv_loss  + mse_loss(out,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(
            f'\rtrain loss : {loss.item():.5f}| step :{epoch}/{opt.epoch}|lr :{lr :.7f} |time_used :{(end - start)  :}',end='', flush=True)
        if (epoch+1) % 50 == 0:
            print('\n ----------------------------------------test!-----------------------------------------------------------')
            with torch.no_grad():
                ssim_eval, psnr_eval = test(net, loader_test)
                print(f'\nstep :{epoch} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')
                ssims.append(ssim_eval)
                psnrs.append(psnr_eval)
                if ssim_eval > max_ssim and psnr_eval > max_psnr:
                    max_ssim = max(max_ssim, ssim_eval)
                    max_psnr = max(max_psnr, psnr_eval)
                # torch.save(net.state_dict(),opt.model_dir+'/train_model.pth')
                torch.save(net,f'../trained_moudles/ll{epoch}.pth')
                list = [epoch, ssim_eval, psnr_eval ]
                data = pd.DataFrame([list])
                data.to_csv('./result.csv',mode='a')
                # print(opt.model_dir+'/train_model.pth')
                print(f'\n model saved at step :{epoch}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

def test(net,loader_test):
    net.eval()

    torch.cuda.empty_cache()
    ssims = []
    psnrs = []

def test(net,loader_test):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    #inputs, targets = next(iter(loader_test))
    for i in range(100):
        for idx,data in enumerate(loader_test,1):
            inputs,targets =data[0],data[1]
            inputs = inputs.to(opt.device)
            targets = targets.to(opt.device)
            pred = net(inputs)  # 预测值
            ssim1 = ssim(pred, targets).item()  # 真实值
            psnr1 = psnr(pred, targets)
            ssims.append(ssim1)
            psnrs.append(psnr1)
            toPIL = transforms.ToPILImage()
            img = pred[0]
            # print(img.shape)
            img1 = targets[0]
            pic = toPIL(img)
            pic1 = toPIL(img1)
            pic.save(save_test_path+f'pre{idx}.jpg')
            pic1.save('clear.jpg')
        return np.mean(ssims), np.mean(psnrs)



# class L1_Charbonnier_loss(torch.nn.Module):
#     """L1 Charbonnierloss."""
#     def __init__(self):
#         super(L1_Charbonnier_loss, self).__init__()
#         self.eps = 1e-6
#
#     def forward(self, X, Y):
#         diff = torch.add(X, -Y)
#         error = torch.sqrt(diff * diff + self.eps)
#         loss = torch.mean(error)
#         return loss


#TEST!
if __name__ == "__main__":

    torch.cuda.empty_cache()
    loader_train = loaders[opt.trainset]    # its_train
    loader_test = loaders[opt.testset]      # its_test
    net = model[opt.net]
    net = net.to(opt.device)
    """
    LOSS
    """
    L1_criterion = nn.L1Loss()
    TV_loss = TVLoss()
    mse_loss = torch.nn.MSELoss()
    ssim = pytorch_ssim.SSIM()
    if torch.cuda.is_available():
        mse_loss = mse_loss.cuda()
        L1_criterion = L1_criterion.cuda()
        TV_loss = TV_loss.cuda()
        ssim = ssim.cuda()


    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.init_lr, betas=(0.9, 0.999),
                            eps=1e-08)
    optimizer.zero_grad()
    train(loader_train,loader_test,net,optimizer)