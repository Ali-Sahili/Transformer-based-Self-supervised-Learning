import math
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

from timm.models.layers import to_2tuple, trunc_normal_

from model.layers import Block, PatchEmbed



class VisionTransformer_SiT(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, 
                       embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
                       qk_scale=None, representation_size=None, distilled=False,
                       drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                       embed_layer=PatchEmbed, norm_layer=None,
                       act_layer=None, weight_init='', training_mode='SSL'):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer 
                                                 (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim  
        self.training_mode=training_mode
        self.num_tokens = 2
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        
        self.rot_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.contrastive_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        self.blocks = nn.Sequential(*[ Block(dim=embed_dim, num_heads=num_heads, 
                                             mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                             qk_scale=qk_scale, drop=drop_rate, 
                                             attn_drop=attn_drop_rate, drop_path=dpr[i], 
                                             norm_layer=norm_layer, act_layer=act_layer)
                                     for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            
            self.pre_logits_rot = nn.Sequential(OrderedDict([
                    ('fc', nn.Linear(embed_dim, representation_size)),
                    ('act', nn.Tanh())
                ]))
            self.pre_logits_contrastive = nn.Sequential(OrderedDict([
                    ('fc', nn.Linear(embed_dim, representation_size)),
                    ('act', nn.Tanh())
                ]))
        else:
            self.pre_logits_rot = nn.Identity()
            self.pre_logits_contrastive = nn.Identity()

        # Classifier head(s)
        if training_mode == 'SSL':
            self.rot_head = nn.Linear(self.num_features, 4) 
            self.contrastive_head = nn.Linear(self.num_features, 512) 
            self.convTrans = nn.ConvTranspose2d(embed_dim, in_chans, 
                                                kernel_size=(patch_size, patch_size), 
                                                stride=(patch_size, patch_size))
            
            # create learnable parameters for the MTL task
            self.rot_w = nn.Parameter(torch.tensor([1.0]))
            self.contrastive_w = nn.Parameter(torch.tensor([1.0]))
            self.recons_w = nn.Parameter(torch.tensor([1.0]))
        else:
            self.rot_head = nn.Linear(self.num_features, num_classes) 
            self.contrastive_head = nn.Linear(self.num_features, num_classes) 
        
        
        # Weight init
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.rot_token, std=.02)
        trunc_normal_(self.contrastive_token, std=.02)

        self.apply(self._init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        self._init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'rot_token', 'contrastive_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if self.training_mode == 'finetune':
            self.rot_head = nn.Linear(self.num_features, num_classes) 
            self.contrastive_head = nn.Linear(self.num_features, num_classes) 
        else:
            self.rot_head = nn.Linear(self.num_features, 4) 
            self.contrastive_head = nn.Linear(self.num_features, 512) 

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        rot_token = self.rot_token.expand(B, -1, -1) 
        contrastive_token = self.contrastive_token.expand(B, -1, -1) 
        x = torch.cat((rot_token, contrastive_token, x), dim=1)
    
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        
        x_rot = self.pre_logits_rot(x[:, 0])
        x_rot = self.rot_head(x_rot)
        
        x_contrastive = self.pre_logits_contrastive(x[:, 1])
        x_contrastive = self.contrastive_head(x_contrastive)

        if self.training_mode == 'finetune':
            return x_rot, x_contrastive
    
        x_rec = x[:, 2:].transpose(1, 2)
        x_rec = self.convTrans(x_rec.unflatten(2, to_2tuple(int(math.sqrt(x_rec.size()[2])))))
        
        return x_rot, x_contrastive, x_rec, self.rot_w, self.contrastive_w, self.recons_w


    def _init_vit_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

