import torch
from torchvision import transforms
from config import config





image_size = config.get('image_size', 128)

import torchvision.transforms as transforms

image_size = 32  # CIFAR-10 image size

# Shared color distortion function
def get_color_distortion(s=0.5):
    color_jitter = transforms.ColorJitter(
        brightness=0.8 * s,
        contrast=0.8 * s,
        saturation=0.8 * s,
        hue=0.2 * s
    )
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

# View 1 
view1_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),  # random crop
    transforms.RandomHorizontalFlip(p=0.5),                           # random flip
    get_color_distortion(s=0.5),                                      # color jitter + grayscale
    transforms.ToTensor(),                                            # to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),           # normalize
])

# View 2
view2_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    get_color_distortion(s=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])









'''
# View 1 random crop + small rotation
view1_transform = transforms.Compose([

    transforms.Resize((image_size, image_size)),
    transforms.RandomResizedCrop(size=image_size, scale=(0.9, 1.0)), # 10% crop change with the image change 
    transforms.ToTensor(),
    transforms.RandomRotation(degrees=10), # 10 dgree 

    
])

# View 2 mild Gaussian noise + intensity / contrast jitter
view2_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),       # mild intensity
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.01*torch.randn_like(x)),  # mild Gaussian noise
])
'''

class SimCLRViewGenerator:
    """

    view 1: crop + rotation
    view 2: noise + intensity jitter
    """
    def __init__(self, view1, view2):
        self.view1 = view1
        self.view2 = view2

    def __call__(self, x):
        return [self.view1(x), self.view2(x)]
