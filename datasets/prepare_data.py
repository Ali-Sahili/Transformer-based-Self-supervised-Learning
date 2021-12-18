import os

from datasets.transforms import build_transform
from datasets.TinyImageNet import TinyImageNetDataset


def build_dataset(args):
    transform = build_transform(False, args)

    root_dir = os.path.join(args.dataset_location, 'tiny-imagenet-200/')
    dataset = TinyImageNetDataset(root_dir=root_dir, mode='val', transform=transform, 
                                  training_mode = args.training_mode,
                                  num_imgs_per_cat=args.num_imgs_per_cat)
    nb_classes = 200


    return dataset, nb_classes

