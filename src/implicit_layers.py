from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import ConvertImageDtype
import torchvision.models.detection as detection

import copy

from dataloader import CocoDataset

import pandas as pd
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class ImplicitAdd(nn.Module):
    def __init__(self, dim):
        super(ImplicitAdd, self).__init__()
        self.dim = dim
        self.implicit = nn.Parameter(torch.zeros(*dim))
        nn.init.normal_(self.implicit, std=.02)

    def forward(self):
        return self.implicit

class ImplicitMul(nn.Module):
    def __init__(self, dim):
        super(ImplicitMul, self).__init__()
        self.dim = dim
        self.implicit = nn.Parameter(torch.ones(*dim))
        nn.init.normal_(self.implicit, mean=1., std=.02)

    def forward(self):
        return self.implicit

class ImplicitWrapperAdd(nn.Module):
    def __init__(self, module, dim):
        super(ImplicitWrapperAdd, self).__init__()
        self.implicit = ImplicitAdd(dim)
        self.module = module

    def forward(self, *x):
        output = self.module(*x)
        return self.implicit().expand_as(output) + output

class ImplicitWrapperMul(nn.Module):
    def __init__(self, module, dim):
        super(ImplicitWrapperMul, self).__init__()
        self.implicit = ImplicitMul(dim)
        self.module = module

    def forward(self, *x):
        output = self.module(*x)
        return self.implicit().expand_as(output) * output