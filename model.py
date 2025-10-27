import torch 
import torch.nn as nn 
from torchvision import models

class model_1(nn.Module):

    def __init__(self,):
        super().__init__()

        self.resnet = models.resnet18()

        # the dim of the mlp the last layer of the resnet for making the attention head 
        mlp_dim = self.resnet.fc.in_features

        # replace the last layer of the reset woth the attention head (we don't need the last layer which ist just prodictoin layer for the resnet)
        # we are using it as a encoder 
        self.resnet.fc = nn.Sequential(nn.Linear(mlp_dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, 128)) # latent space dim = 128 defult of the paper 

    def forward(self,x):
        return self.resnet(x)


    


