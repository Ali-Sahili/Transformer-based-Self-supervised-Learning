import torch
import kornia
import numpy as np
from numpy.random import randint



def drop_rand_patches(X, X_rep=None, max_drop=0.3, max_block_sz=0.25, tolr=0.05):
    #######################
    # X_rep: replace X with patches from X_rep. If X_rep is None, replace the patches with Noise
    # max_drop: percentage of image to be dropped
    # max_block_sz: percentage of the maximum block to be dropped
    # tolr: minimum size of the block in terms of percentage of the image size
    #######################
    
    C, H, W = X.size()
    n_drop_pix = np.random.uniform(0, max_drop)*H*W
    mx_blk_height = int(H*max_block_sz)
    mx_blk_width = int(W*max_block_sz)
    
    tolr = (int(tolr*H), int(tolr*W))
    
    total_pix = 0
    while total_pix < n_drop_pix:
        
        # get a random block by selecting a random row, column, width, height
        rnd_r = randint(0, H-tolr[0])
        rnd_c = randint(0, W-tolr[1])
        # rnd_r is alread added - this is not height anymore
        rnd_h = min(randint(tolr[0], mx_blk_height)+rnd_r, H) 
        rnd_w = min(randint(tolr[1], mx_blk_width)+rnd_c, W)
        
        if X_rep is None:
            X[:, rnd_r:rnd_h, rnd_c:rnd_w] = torch.empty((C, rnd_h-rnd_r, rnd_w-rnd_c), dtype=X.dtype, device='cuda').normal_()
        else:
            X[:, rnd_r:rnd_h, rnd_c:rnd_w] = X_rep[:, rnd_r:rnd_h, rnd_c:rnd_w]    
         
        total_pix = total_pix + (rnd_h-rnd_r)*(rnd_w-rnd_c)

    return X

def rgb2gray_patch(X, tolr=0.05):

    C, H, W = X.size()
    tolr = (int(tolr*H), int(tolr*W))
     
    # get a random block by selecting a random row, column, width, height
    rnd_r = randint(0, H-tolr[0])
    rnd_c = randint(0, W-tolr[1])
    rnd_h = min(randint(tolr[0], H)+rnd_r, H) #rnd_r is alread added - this is not height anymore
    rnd_w = min(randint(tolr[1], W)+rnd_c, W)
    
    X[:, rnd_r:rnd_h, rnd_c:rnd_w] = torch.mean(X[:, rnd_r:rnd_h, rnd_c:rnd_w], dim=0).unsqueeze(0).repeat(C, 1, 1)

    return X
    

def smooth_patch(X, max_kernSz=15, gauss=5, tolr=0.05):

    #get a random kernel size (odd number)
    kernSz = 2*(randint(3, max_kernSz+1)//2)+1
    gausFct = np.random.rand()*gauss + 0.1 # generate a real number between 0.1 and gauss+0.1
    
    C, H, W = X.size()
    tolr = (int(tolr*H), int(tolr*W))
     
    # get a random block by selecting a random row, column, width, height
    rnd_r = randint(0, H-tolr[0])
    rnd_c = randint(0, W-tolr[1])
    rnd_h = min(randint(tolr[0], H)+rnd_r, H) #rnd_r is alread added - this is not height anymore
    rnd_w = min(randint(tolr[1], W)+rnd_c, W)
    
    
    gauss = kornia.filters.GaussianBlur2d((kernSz, kernSz), (gausFct, gausFct))
    X[:, rnd_r:rnd_h, rnd_c:rnd_w] = gauss(X[:, rnd_r:rnd_h, rnd_c:rnd_w].unsqueeze(0))
    
    return X

def distortImages(samples):
    # this is batch size, but in case bad inistance happened while loading
    n_imgs = samples.size()[0] 
    samples_aug = samples.detach().clone()
    for i in range(n_imgs):

        samples_aug[i] = rgb2gray_patch(samples_aug[i])

        samples_aug[i] = smooth_patch(samples_aug[i])

        samples_aug[i] = drop_rand_patches(samples_aug[i])

        idx_rnd = randint(0, n_imgs)
        if idx_rnd != i:
            samples_aug[i] = drop_rand_patches(samples_aug[i], samples_aug[idx_rnd])
      
    return samples_aug
