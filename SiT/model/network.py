import math
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn

from model.transformer import Transformer
from model.layers import Block, PatchEmbed
from timm.models.layers import to_2tuple, trunc_normal_



""" Original implementation of SiT architecture """
class SiT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, 
                       embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
                       qk_scale=None, representation_size=None,
                       drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                       embed_layer=PatchEmbed, training_mode='SSL'):
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
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
        """
        super().__init__()
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim  
        self.training_mode=training_mode
        self.num_tokens = 2
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

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
                                             attn_drop=attn_drop_rate, drop_path=dpr[i])
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
        self._init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'rot_token', 'contrastive_token'}

    def features_extraction(self, x):
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
        x = self.features_extraction(x)
        
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


""" Another implementation of SiT architecture """
class SiT_2(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, 
                       rotation_node=4, contrastive_head=512, 
                       dim=768, depth=12, heads=12, in_channels=3, dim_head=64, 
                       dropout=0., emb_dropout=0., scale_dim=4, training_mode = 'SSL'):
        super().__init__()

        assert image_size%patch_size == 0, 'Image dimensions must be divisible by patch size.'
                
        self.training_mode = training_mode
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        h = image_size // patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, dim))
        self.rotation_token = nn.Parameter(torch.randn(1, 1, dim))
        self.contrastive_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)
        

        # Classifier head(s)
        if training_mode == 'SSL':
            self.rot_head = nn.Sequential(nn.LayerNorm(dim),
                                          nn.Linear(dim, rotation_node)
                                         ) 
            self.contrastive_head = nn.Sequential( nn.LayerNorm(dim),
                                                   nn.Linear(dim, contrastive_head)
                                                 )
            self.to_img = nn.Sequential( nn.Linear(dim, patch_dim),
                                     Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                                                  h=h, w=h, p1=patch_size, p2=patch_size, 
                                                  c=in_channels)
                                   )
            
            # create learnable parameters for the MTL task
            self.rot_w = nn.Parameter(torch.tensor([1.0]))
            self.contrastive_w = nn.Parameter(torch.tensor([1.0]))
            self.recons_w = nn.Parameter(torch.tensor([1.0]))
        else:
            self.rot_head = nn.Sequential(nn.LayerNorm(dim),
                                          nn.Linear(dim, num_classes)
                                         ) 
            self.contrastive_head = nn.Sequential( nn.LayerNorm(dim),
                                                   nn.Linear(dim, num_classes)
                                                 )

        # Weight init
        trunc_normal_(self.pos_embedding, std=.02)
        trunc_normal_(self.rotation_token, std=.02)
        trunc_normal_(self.contrastive_token, std=.02)

        self.apply(self._init_vit_weights)

    def _init_vit_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        rotation_tokens = repeat(self.rotation_token, '() n d -> b n d', b=b)
        contrastive_tokens = repeat(self.contrastive_token, '() n d -> b n d', b=b)
        x = torch.cat((rotation_tokens, contrastive_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 2)]
        x = self.dropout(x)

        x = self.transformer(x)

        l_rot = self.rot_head(x[:, 0])
        l_contrastive = self.contrastive_head(x[:, 1])

        if self.training_mode == 'finetune':
            return l_rot, l_contrastive
    
        x_recons = self.to_img(x[:, 2:])
        
        return l_rot, l_contrastive, x_recons, self.rot_w, self.contrastive_w, self.recons_w

