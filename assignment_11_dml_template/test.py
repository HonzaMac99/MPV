import torch, torchvision
from torchvision import transforms
from torchvision.models import ResNet18_Weights
import torch.optim as optim, torch.nn as nn, torch.nn.functional as F, torch.utils.data as data
import os, random, copy, pdb, numpy as np, ssl
from PIL import Image
from scipy.spatial.distance import cdist
from numpy import loadtxt

ssl._create_default_https_context = ssl._create_unverified_context # fix needed for downloading resnet weights

from dml import *

# device = torch.device('cuda:0') # gpu training is typically much faster if available
# device = torch.device('cpu')

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

pre_model = torchvision.models.resnet18(weights = ResNet18_Weights.DEFAULT) # load pre-trained ResNet18
dim = 512  ## dimensionality of the global descriptor - defined according to the ResNet18 architecture
model = GDextractor(pre_model, dim) # construct the network that extracts descriptors

...