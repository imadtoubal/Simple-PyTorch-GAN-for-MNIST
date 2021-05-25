import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)
    
    
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.layers(x)
