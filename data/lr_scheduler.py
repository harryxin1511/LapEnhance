import math
# 余弦退火学习率
from data.option import opt




def lr_schedule_cosdecay(t, T, init_lr=opt.init_lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr