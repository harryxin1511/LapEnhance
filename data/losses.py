import torch
import torch.nn as nn



class LaplaceLoss(nn.Module):
    pass



class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

L1_criterion = nn.L1Loss()
TV_loss = TVLoss()
mse_loss = torch.nn.MSELoss()
ssim = pytorch_ssim.SSIM()
if cuda:
    gpus_list = range(opt.gpus)
    mse_loss = mse_loss.cuda()
    L1_criterion = L1_criterion.cuda()
    TV_loss = TV_loss.cuda()
    ssim = ssim.cuda(gpus_list[0])