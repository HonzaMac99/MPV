import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kornia as K
import typing
from typing import Tuple, List, Dict
from PIL import Image
import os
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from time import time
import pickle
import copy

# -------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torchvision as tv
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
# -------------------------------------------------------------------------------------------------------

######################################################################
#                             Macros                                 #
######################################################################
TQDM = True
VGG16 = True
LOAD = True
TRAIN = False
SAVE = False


# wrapper for get_dataset_statistics()
def get_stats(dataset: torch.utils.data.Dataset) -> Tuple[List, List]:
    if os.path.isfile("stats.pkl"):
        print("Loading statistics...")
        with open("stats.pkl", "rb") as f:
            stats = pickle.load(f)
        mean, std = stats
    else:
        print("Computing statistics...")
        mean, std = get_dataset_statistics(ImageNette_for_statistics)
        stats = [mean, std]
        with open("stats.pkl", "wb") as f:
            pickle.dump(stats, f)
    return mean, std


def get_dataset_statistics(dataset: torch.utils.data.Dataset) -> Tuple[List, List]:
    '''Function, that calculates mean and std of a dataset (pixelwise)
    Return:
        tuple of Lists of floats. len of each list should equal to number of input image/tensor channels
    '''
    c, h, w = dataset[0][0].shape

    tensors = [tuple[0] for tuple in dataset]
    data = torch.stack(tensors)

    mean_arr = []
    std_arr = []
    for i in range(c):
        mean_arr.append(data[:, i, ...].mean().item())
        std_arr.append(data[:, i, ...].std().item())

    return mean_arr, std_arr


class SimpleCNN_orig(nn.Module):
    """Class, which implements image classifier. """
    def __init__(self, num_classes = 10):
        super(SimpleCNN_orig, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride = 2, padding=3, bias = False),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, kernel_size=5, stride = 2, padding=2, bias = False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, kernel_size=3, stride = 1, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

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

        # clasifier
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


# VGG-16
class SimpleCNN(nn.Module):
    """Class, which implements image classifier. """
    def __init__(self, num_classes = 10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # conv block 1 (2x)
            nn.Conv2d(3, 64, kernel_size=3, stride = 2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # [224x224] ---> [112x112]

            # conv block 2 (2x)
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # [112x112] ---> [56x56]

            # conv block 3 (3x)
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # [56x56] ---> [28x28]

            # conv block 4 (3x)
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # [28x28] ---> [14x14]

            # conv block 5 (3x)
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # [14x14] ---> [7x7]
            nn.Dropout(0.1)
        )

        # clasifier
        ll_units = 4096
        self.clf = nn.Sequential(nn.AdaptiveAvgPool2d((7, 7)),
                                 nn.Flatten(),

                                 # linear 1
                                 nn.Linear(7*7*512, ll_units),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(),

                                 # linear 2
                                 nn.Linear(ll_units, ll_units),
                                 nn.BatchNorm1d(ll_units),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(),

                                 # linear 3
                                 nn.Linear(ll_units, num_classes))
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
        # m.weight.data.uniform_(0.0, 1.0)
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
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
    '''
    Function, which runs training over a single epoch in the dataloader and returns the model.
    Do not forget to set the model into train mode and zero_grad() optimizer before backward.
    '''
    model.train()

    if epoch_idx == 0:
        val_loss, additional_out = validate(model, val_loader, loss_fn, device, additional_params)
        print("init accuracy: {:.2f}% | init loss: {:.2f}".format(additional_out['acc']*100, val_loss))
        model = model.to(device)
        # TODO: try tensorboaard?
        # if writer is not None:
        #     # if do_acc:
        #     if 'with_acc' in additional_params and additional_params['with_acc']:
        #         writer.add_scalar("Accuracy/val", additional_out['acc'], 0)
        #     writer.add_scalar("Loss/val", val_loss, 0)

    acc = 0
    running_loss = 0
    n_samples = 0
    iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc="training") if TQDM else enumerate(train_loader)
    for idx, (data, labels) in iterator:
        optim.zero_grad()  # reset all gradients of all torch tensors

        pred = model(data)
        n_samples += pred.shape[0]
        loss = loss_fn(pred, labels)

        loss.backward()
        optim.step()

        running_loss += loss.item()

        pred_y = torch.argmax(pred, dim=1)
        acc += torch.sum(pred_y == labels).item()

        if not TQDM and idx % 50 == 0:
            print(f"Epoch: {epoch_idx} \t Iter: {idx + 1} / {int(len(train_loader.dataset) / train_loader.batch_size) + 1}"
                  f" \t loss: {running_loss / (idx + 1) :.3f}")
    print(f"Epoch: {epoch_idx} \t loss: {running_loss / len(train_loader) :.3f}")

    loss, acc = validate(model, val_dl, loss_function)
    print("accuracy: {:.2f}% | loss: {:.2f}\n".format(acc['acc']*100, loss))
    return model


# wrapper for lr_find()
def get_best_lr(model: torch.nn.Module,
                train_dl:torch.utils.data.DataLoader,
                loss_fn:torch.nn.Module,
                min_lr: float=1e-7, max_lr:float=100, steps:int = 50, max_iter:int = 20)-> float:

    losses, lrs = lr_find(model, train_dl, loss_function, min_lr=1e-7, max_lr=1, steps=10, max_iter=5)
    best_lr = lrs[np.argmin(losses)]
    print("Best lr so far: ", best_lr)

    losses, lrs = lr_find(model, train_dl, loss_function, min_lr=best_lr/10, max_lr=best_lr*10, steps=10, max_iter=5)
    best_lr = lrs[np.argmin(losses)]
    print("Best lr: ", best_lr)
    return best_lr


def lr_find(model: torch.nn.Module,
            train_dl:torch.utils.data.DataLoader,
            loss_fn:torch.nn.Module,
            min_lr: float=1e-7, max_lr:float=100, steps:int = 50, max_iter:int = 20)-> Tuple:
    '''
    Function, which run the training for a small number of iterations, increasing the learning rate
    and storing the losses. Model initialization is saved before training and restored after training
    '''
    model.train()

    min_lr_log = torch.log10(torch.tensor(min_lr)).item()
    max_lr_log = torch.log10(torch.tensor(max_lr)).item()

    lrs = np.logspace(min_lr_log, max_lr_log, steps)
    losses = np.zeros_like(lrs)
    init_weights = model.state_dict()  # save the model initialization
    for i, lr in enumerate(lrs):
        model.load_state_dict(init_weights)  # restore the model initialization
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        running_loss = 0
        iterator = tqdm(enumerate(train_dl), total=max_iter+1, desc="testing lr") if TQDM else enumerate(train_dl)
        for idx, (data, labels) in iterator:
            optimizer.zero_grad()  # reset all gradients of all torch tensors
            pred = model(data)

            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if idx >= max_iter:
                break
        print("lr: {:.2e} | loss: {:.2f}".format(lr, running_loss / max_iter))
        losses[i] = running_loss / max_iter

    return losses, lrs


def validate(model: torch.nn.Module,
             val_dl: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device = torch.device('cpu'),
             additional_params: Dict = {}) -> Tuple[float, Dict]:
    '''
    Function, which runs the module over validation set and returns accuracy
    '''
    model.eval()

    # do_acc = False
    # if 'with_acc' in additional_params:
    #     do_acc = additional_params['with_acc']

    acc = 0
    loss = 0
    n_samples = 0
    iterator = tqdm(enumerate(val_dl), total=len(val_dl), desc="validating") if TQDM else enumerate(val_dl)
    for idx, (data, labels) in iterator:
        with torch.no_grad():
            batch_output = model(data)
        loss += loss_fn(batch_output, labels).item()

        batch_preds = torch.argmax(batch_output, dim=1)
        acc += torch.sum(batch_preds == labels).item()
        n_samples += batch_output.shape[0]

    acc = acc / n_samples
    return loss, {'acc': acc}


class TestFolderDataset(torch.utils.data.Dataset):
    '''Class, which reads images in folder and serves as test dataset'''

    def __init__(self, folder_name, transform = None):
        self.files = os.listdir(folder_name)
        self.folder_name = folder_name
        self.transform = transform
        return

    def get_fname(self, idx):
        return os.path.join(self.folder_name, self.files[idx])

    def __getitem__(self, index):
        img = Image.open(self.get_fname(index))
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        ln = len(self.files)
        return ln
        

def get_predictions(model: torch.nn.Module, test_dl: torch.utils.data.DataLoader) -> torch.Tensor:
    '''
    Function, which predicts class indexes for image in data loader.
    Ouput shape: [N, 1], where N is number of image in the dataset
    '''

    out = torch.zeros(0).long()
    for idx, (data, labels) in enumerate(test_dl):
        with torch.no_grad():
            batch_output = model(data)   # 32x10
        batch_preds = torch.argmax(batch_output, dim=1)
        out = torch.cat((out, batch_preds), 0)
    return out


if __name__ == "__main__":
    resolution = 224 if VGG16 else 128
    base_tf = tfms.Compose([tfms.Resize((resolution, resolution)), tfms.ToTensor()])
    ImageNette_for_statistics = tv.datasets.ImageFolder('imagenette2-160/train', transform=base_tf)
    print("Number of images: ", len(ImageNette_for_statistics))

    mean, std = get_stats(ImageNette_for_statistics)
    print("Mean: ", mean)  # [0.46248055, 0.4579692, 0.42981696]
    print("Std: ", std)    # [0.27553096, 0.27220666, 0.295335]

    # include normalisation
    train_tf = tfms.Compose([tfms.Resize((resolution, resolution)),
                             tfms.RandomHorizontalFlip(p=0.5),
                             tfms.RandomResizedCrop(size=(resolution, resolution)),
                             tfms.ToTensor(),
                             tfms.Normalize(mean, std)])

    val_tf = tfms.Compose([tfms.Resize((resolution, resolution)),
                           tfms.ToTensor(),
                           tfms.Normalize(mean, std)])

    ImageNette_train = tv.datasets.ImageFolder('imagenette2-160/train', transform=train_tf)
    ImageNette_val = tv.datasets.ImageFolder('imagenette2-160/val', transform=val_tf)

    num_workers = os.cpu_count()
    if 'sched_getaffinity' in dir(os):
        num_workers = len(os.sched_getaffinity(0)) - 2

    batch_size = 32  # dl. lengths match the number of images in coresp. folders
    train_dl = torch.utils.data.DataLoader(ImageNette_train,
                                           batch_size= batch_size,
                                           shuffle = True, # important thing to do for training.
                                           num_workers = num_workers)
    val_dl = torch.utils.data.DataLoader(ImageNette_val,
                                         batch_size= batch_size,
                                         shuffle = False,
                                         num_workers = num_workers,
                                         drop_last=False)  # do not drop the last (smaller) batch for validation
    # batch.shape: [batch_size, 3, resolution, resolution]

    num_classes = len(ImageNette_train.classes)
    learning_rate = 0.001
    weight_decay = 1e-6
    epochs = 20
    model = SimpleCNN(num_classes) if VGG16 else SimpleCNN_orig(num_classes)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = torch.nn.CrossEntropyLoss()
    print("Num_workers: ", num_workers)
    print("Batch_size: ", batch_size)
    print("Resolution: ", resolution)
    print("Learning_rate: ", learning_rate)
    print("Weight_decay: ", weight_decay)
    print("Epochs: ", epochs)

    if LOAD:
        print("Loading weights...")
        model.load_state_dict(torch.load('weights_vgg16light.pts'))
    else:
        print("Initializing weights...")
        model.features.apply(weight_init)
        model.clf.apply(weight_init)

    if TRAIN:
        print("Looking for best lr...")
        # vgg16light: 0.0016681005372000575
        best_lr = get_best_lr(model, train_dl, loss_function)
        optimizer = torch.optim.SGD(model.parameters(), lr=best_lr, momentum=0.9)

        print("Training...")
        for i in range(epochs):   # one epoch = 30 min for VGG16, good luck
            model = train_and_val_single_epoch(model, train_dl, val_dl, optimizer, loss_function, i)
            if epochs % 10 == 0:
                best_lr = get_best_lr(model, train_dl, loss_function)
                optimizer = torch.optim.SGD(model.parameters(), lr=best_lr, momentum=0.9)
        print("Done.")

    if SAVE:
        print("Saving weights...")
        torch.save(model.state_dict(), 'weights.pts')

    print("Testing...")
    names = ["tench", "English springer", "cassette player", "chain saw", "church",
             "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]
    preds = []
    model.eval()
    test_dl = torch.utils.data.DataLoader(TestFolderDataset('test_set', val_tf))
    display_dl = torch.utils.data.DataLoader(TestFolderDataset('test_set', base_tf))
    for i, (data, img) in enumerate(zip(test_dl, display_dl)):
        with torch.no_grad():
            logits = model(data)
        y_pred = torch.argmax(logits, dim=1).item()
        preds.append(y_pred)

        torch.set_printoptions(linewidth=300)
        # print(logits)
        print(f"Img no. {i+1}: class {y_pred} --> {names[y_pred]}")
        # imshow_torch(img)
        # plt.show()
    print("Done.")

    import csv
    csv_file = "submission.csv"
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        for y in preds:
            writer.writerow([y])

