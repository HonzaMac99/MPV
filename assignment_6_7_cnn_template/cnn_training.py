import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kornia as K
import typing
from typing import Tuple, List, Dict
from PIL import Image
import os
from tqdm import tqdm_notebook as tqdm
from time import time


def get_dataset_statistics(dataset: torch.utils.data.Dataset) -> Tuple[List, List]:
    '''Function, that calculates mean and std of a dataset (pixelwise)
    Return:
        tuple of Lists of floats. len of each list should equal to number of input image/tensor channels
    '''

    mean_arr = []
    std_arr = []

    b = len(dataset)
    c, h, w = dataset[0][0].shape
    for i in range(b):
        mean_i = [0., 0., 0.]
        std_i = [1.0, 1.0, 1.0]
        for j in range(c):
            mean_i[j] = dataset[i][0][j].mean()
            std_i[j] = dataset[i][0][j].std()
        mean_arr.append(mean_i)
        std_arr.append(std_i)

    mean = torch.Tensor(mean_arr).mean(axis=0)
    std = torch.Tensor(std_arr).mean(axis=0)

    return mean.tolist(), std.tolist()


class SimpleCNN(nn.Module):
    """Class, which implements image classifier. """
    def __init__(self, num_classes = 10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride = 2, padding=3, bias = False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride = 2, padding=2, bias = False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride = 1, padding=1, bias = False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride = 1, padding=1, bias = False),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride = 2, padding=1, bias = False),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride = 2, padding=1, bias = False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.clf = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Flatten(),
                                nn.Linear(512, num_classes))
        return
    def forward(self, input):
        """ 
        Shape:
        - Input :math:`(B, C, H, W)` 
        - Output: :math:`(B, NC)`, where NC is num_classes
        """
        x = self.features(input)
        return self.clf(x)


def weight_init(m: nn.Module) -> None:
    '''Function, which fills-in weights and biases for convolutional and linear layers'''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #do something here. You can access layer weight or bias by m.weight or m.bias
     pass #do something
    return

def train_and_val_single_epoch(model: torch.nn.Module,
                       train_loader: torch.utils.data.DataLoader,
                       val_loader: torch.utils.data.DataLoader,
                       optim: torch.optim.Optimizer,
                       loss_fn: torch.nn.Module,
                       epoch_idx = 0,
                       lr_scheduler = None,
                       writer = None,
                       device: torch.device = torch.device('cpu'),
                       additional_params: Dict = {}) -> torch.nn.Module:
    '''Function, which runs training over a single epoch in the dataloader and returns the model. Do not forget to set the model into train mode and zero_grad() optimizer before backward.'''
    model.train()
    if epoch_idx == 0:
        val_loss, additional_out = validate(model, val_loader, loss_fn, device, additional_params)
        model = model.to(device)
        if writer is not None:
            if do_acc:
                writer.add_scalar("Accuracy/val", additional_out['acc'], 0)
            writer.add_scalar("Loss/val", val_loss, 0)
    for idx, (data, labels) in tqdm(enumerate(train_loader), total=num_batches):
         pass #do something
    return model

def lr_find(model: torch.nn.Module,
            train_dl:torch.utils.data.DataLoader,
            loss_fn:torch.nn.Module,
            min_lr: float=1e-7, max_lr:float=100, steps:int = 50)-> Tuple:
    '''Function, which run the training for a small number of iterations, increasing the learning rate and storing the losses. Model initialization is saved before training and restored after training'''
    lrs = np.ones(steps)
    losses = np.ones(steps)
    return losses, lrs


def validate(model: torch.nn.Module,
             val_loader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device = torch.device('cpu'),
             additional_params: Dict = {}) -> Tuple[float, Dict]:
    '''Function, which runs the module over validation set and returns accuracy'''
    print("Starting validation")
    acc = 0
    loss = 0
    do_acc = False
    if 'with_acc' in additional_params:
        do_acc = additional_params['with_acc']
    for idx, (data, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
        with torch.no_grad():
             pass #do something
    return loss, {'acc': acc}



class TestFolderDataset(torch.utils.data.Dataset):
    '''Class, which reads images in folder and serves as test dataset'''
    def __init__(self, folder_name, transform = None):
        return
    def __getitem__(self, index):
        img = Image.new('RGB', (128, 128))
        return img
    def __len__(self):
        ln = 0
        return ln
        

def get_predictions(model: torch.nn.Module, test_dl: torch.utils.data.DataLoader)->torch.Tensor :
    '''Function, which predicts class indexes for image in data loader. Ouput shape: [N, 1], where N is number of image in the dataset'''
    out = torch.zeros(len(test_dl)).long()
    return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    import numpy as np
    import cv2

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision as tv
    import kornia as K
    from tqdm import tqdm_notebook as tqdm
    from time import time
    import torchvision.transforms as tfms


    def imshow_torch(tensor, figsize=(8, 6), *kwargs):
        plt.figure(figsize=figsize)
        plt.imshow(K.tensor_to_image(tensor), *kwargs)
        return


    def imshow_torch_channels(tensor, dim=1, *kwargs):
        num_ch = tensor.size(dim)
        fig = plt.figure(figsize=(num_ch * 5, 5))
        tensor_splitted = torch.split(tensor, 1, dim=dim)
        for i in range(num_ch):
            fig.add_subplot(1, num_ch, i + 1)
            plt.imshow(K.tensor_to_image(tensor_splitted[i].squeeze(dim)), *kwargs)
        return


    train_transform = tfms.Compose([tfms.Resize((128, 128)),
                                    tfms.ToTensor()])

    ImageNette_for_statistics = tv.datasets.ImageFolder('imagenette2-160/train',
                                                        transform=train_transform)

    for i in range(3):
        imshow_torch(ImageNette_for_statistics[i][0], figsize=(3, 3))


    mean, std = get_dataset_statistics(ImageNette_for_statistics)
    #mean, std = [0.46248055, 0.4579692, 0.42981696], [0.27553096, 0.27220666, 0.295335]
    print (mean, std)
