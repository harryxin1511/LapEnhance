### step1 在最后的重建加入原始图  
psnr 19.55 x


### 最底层unet去除原图 
20。71 x

### sam feature :32  
psnr 21

#sam feature 9
psnr 20.6


#add vgg loss  all vgg loss epoch 20000 
psnr 22.2

###  loss = scaleloss + 6*ssim_loss
23.49

###   gs filter  loss = scaleloss + ssim_loss

### srm net 4.3
23.7  w/o lapmap in last phase
### 