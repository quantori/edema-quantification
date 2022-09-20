"""Models for the edema classification project.
    
The description to be filled...
"""

from distutils.command.sdist import sdist
import torch
from torch import nn


class SqueezeNet(nn.Module):
    """SqueezeNet backbone.

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        super().__init__()

        model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=True)


        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        sdist dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
        fdfj
        dic = {1:1, 
        
        2:3,  9:0
        
        
        234:5}
        


    def forward(self, x):
        return self.l1(x)