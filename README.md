# Transformer-based-Self-supervised-Learning
Pytorch implementation of two transformer-based networks which learn in a self-supervised manner.

![](SSL.gif)

## Introduction
Over the last few years, self-supervision has taken attention and leads to great improvements comparable to full-supervised modesl or even better in some cases which make it as the next stage in deep learning approaches. Meanwhile, transformers help to build meaningful global-scale connections between embeddings and activation maps for different classes and currently vision transformers are able to take place of conolution layers to achieve the best performance.

Based on the above, combining these two notions could lead to promising results and overcome all difficulties mentioned in the old models. Recently, two different works ([Atito et al.](https://arxiv.org/abs/2104.03602), [Caron et al.](https://arxiv.org/abs/2104.14294)) rely on this methodlogy and show interesting results. This repo supports the implementation of these two methods on Pytorch in a simple manner.

For more details about the context, see this interesting blog [here](https://towardsdatascience.com/self-supervised-learning-in-vision-transformers-30ff9be928c).

## Organization

```
.
├─ SiT/                      <- Implementation of Self-supervised vIsion Transformer
│  └─ ...   
│
├─ DINO/                     <- Implementation of self-DIstillation with NO labels
│  └─ ...      
│
├─ SSL.gif          
└─ README.md
```



## References
1. [SiT: Self-supervised vIsion Transformer](https://arxiv.org/abs/2104.03602)
2. [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
<!-- 3. [Efficient Self-supervised Vision Transformers for Representation Learning](https://arxiv.org/abs/2106.09785) -->
