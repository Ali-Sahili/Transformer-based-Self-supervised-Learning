
from torchvision import transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD



def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        transform = create_transform(input_size=args.input_size,
                                     is_training=True,
                                     color_jitter=args.color_jitter,
                                     auto_augment=args.aa,
                                     interpolation=args.train_interpolation,
                                     re_prob=args.reprob,
                                     re_mode=args.remode,
                                     re_count=args.recount,
                                    )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append( transforms.Resize(size, interpolation=3))
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
