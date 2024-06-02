import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from typing import Tuple
from imagefiltering import * 
from local_detector import *


def affine_from_location(b_ch_d_y_x: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
    r"""Computes transformation matrix A which transforms point in homogeneous coordinates from canonical coordinate system into image
    from keypoint location (output of scalespace_harris or scalespace_hessian)
    Return:
        torch.Tensor: affine tranformation matrix

    Shape:
        - Input :math:`(B, 5)` 
        - Output: :math:`(B, 3, 3)`, :math:`(B, 1)`

    """
    A = torch.zeros(b_ch_d_y_x.size(0), 3, 3)
    img_idxs = torch.zeros(b_ch_d_y_x.size(0), 1).long()

    A[:,0,0] = b_ch_d_y_x[:,2]
    A[:,1,1] = b_ch_d_y_x[:,2]
    A[:,0,2] = b_ch_d_y_x[:,4]
    A[:,1,2] = b_ch_d_y_x[:,3]
    A[:,2,2] = 1

    img_idxs = b_ch_d_y_x[:,0].reshape(-1,1).long()

    return A, img_idxs


def affine_from_location_and_orientation(b_ch_d_y_x: torch.Tensor,
                                         ori: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
    r"""Computes transformation matrix A which transforms point in homogeneous coordinates from canonical coordinate system into image
    from keypoint location (output of scalespace_harris or scalespace_hessian). Ori - orientation angle in radians
    Return:
        torch.Tensor: affine tranformation matrix

    Shape:
        - Input :math:`(B, 5)`, :math:`(B, 1) 
        - Output: :math:`(B, 3, 3)`, :math:`(B, 1)`

    """
    A = torch.zeros(b_ch_d_y_x.size(0), 3, 3)
    img_idxs = torch.zeros(b_ch_d_y_x.size(0), 1).long()

    A, img_idxs = affine_from_location(b_ch_d_y_x)

    R = torch.zeros(b_ch_d_y_x.size(0), 3, 3)
    R[:,0,0] = torch.cos(ori).reshape(-1)
    R[:,1,1] = torch.cos(ori).reshape(-1)
    R[:,0,1] = torch.sin(ori).reshape(-1)
    R[:,1,0] = -torch.sin(ori).reshape(-1)
    R[:,2,2] = 1

    A = torch.matmul(A, R)

    return A, img_idxs


def affine_from_location_and_orientation_and_affshape(b_ch_d_y_x: torch.Tensor,
                                                      ori: torch.Tensor,
                                                      aff_shape: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
    r"""Computes transformation matrix A which transforms point in homogeneous coordinates from canonical coordinate system into image
    from keypoint location (output of scalespace_harris or scalespace_hessian)
    Return:
        torch.Tensor: affine tranformation matrix

    Shape:
        - Input :math:`(B, 5)`, :math:`(B, 1), :math:`(B, 3)
        - Output: :math:`(B, 3, 3)`, :math:`(B, 1)`

    """
    A = torch.zeros(b_ch_d_y_x.size(0), 3, 3)
    img_idxs = torch.zeros(b_ch_d_y_x.size(0), 1).long()

    return A, img_idxs


def estimate_patch_dominant_orientation(x: torch.Tensor, num_angular_bins: int = 36):
    """Function, which estimates the dominant gradient orientation of the given patches, in radians.
    Zero angle points towards right.
    
    Args:
        x: (torch.Tensor) shape (B, 1, PS, PS)
        num_angular_bins: int, default is 36
    
    Returns:
        angles: (torch.Tensor) in radians shape [Bx1]
    """
    out = torch.zeros(x.size(0), 1)
    b, c, h, w = x.shape

    grad = spatial_gradient_first_order(x, 1.0)
    g = torch.unbind(grad, dim=2)
    gx = g[0].squeeze(1)
    gy = g[1].squeeze(1)

    mags = torch.sqrt(torch.pow(gx,2) + torch.pow(gy,2))
    orients = torch.rad2deg(torch.atan2(gy,gx))
    # print(orients.shape)

    histogram = torch.zeros(b,num_angular_bins)
    one_bin = 360/num_angular_bins
    
    center = np.floor(w/2)

    for x in range(w):
        for y in range(h):
            for i in range(b):
                if (x - center)**2 + (y - center)**2 < center**2:
                    # print(orients[i,y,x])
                    # print(one_bin)
                    bin = torch.floor(orients[i,y,x] / one_bin).int()
                    histogram[i,bin] += mags[i,y,x]

    out = torch.deg2rad(torch.argmax(histogram,dim=1) * one_bin)

    return out

def estimate_patch_affine_shape(x: torch.Tensor):
    """Function, which estimates the patch affine shape by second moment matrix. Returns ellipse parameters: a, b, c
    Args:
        x: (torch.Tensor) shape (B, 1, PS, PS)
    
    Returns:
        ell: (torch.Tensor) in radians shape [Bx3]
    """
    out = torch.zeros(x.size(0), 3)
    return out

def calc_sift_descriptor(input: torch.Tensor,
                  num_ang_bins: int = 8,
                  num_spatial_bins: int = 4,
                  clipval: float = 0.2) -> torch.Tensor:
    '''    
    Args:
        x: torch.Tensor (B, 1, PS, PS)
        num_ang_bins: (int) Number of angular bins. (8 is default)
        num_spatial_bins: (int) Number of spatial bins (4 is default)
        clipval: (float) default 0.2
        
    Returns:
        Tensor: SIFT descriptor of the patches

    Shape:
        - Input: (B, 1, PS, PS)
        - Output: (B, num_ang_bins * num_spatial_bins ** 2)
    '''
    out = torch.zeros(input.size(0), num_ang_bins * num_spatial_bins ** 2)
    b, c, h, w = input.shape

    grad = spatial_gradient_first_order(input, 1.0)
    g = torch.unbind(grad, dim=2)
    gx = g[0].squeeze(1)
    gy = g[1].squeeze(1)

    mags = torch.sqrt(torch.pow(gx,2) + torch.pow(gy,2))
    orients = torch.rad2deg(torch.atan2(gy,gx))

    one_bin = 360/num_ang_bins

    spatial_bin = int(np.floor(h/num_spatial_bins))
    center = np.floor(w/2)

    for sb in range(num_spatial_bins**2):
        x_off = int(sb % num_spatial_bins) * spatial_bin
        y_off = int(np.floor(sb / num_spatial_bins)) * spatial_bin
        for x in range(spatial_bin):
            for y in range(spatial_bin):
                for i in range(b):
                    if (x - center + x_off)**2 + (y - center + y_off)**2 < center**2:
                        bin = torch.floor(orients[i,y+y_off,x+x_off] / one_bin).int() + sb*num_ang_bins
                        out[i,bin] += mags[i,y+y_off,x+x_off]

    return out


def photonorm(x: torch.Tensor):
    """Function, which normalizes the patches such that the mean intensity value per channel will be 0 and the standard deviation will be 1.0. Values outside the range < -3,3> will be set to -3 or 3 respectively
    Args:
        x: (torch.Tensor) shape [BxCHxHxW]
    
    Returns:
        out: (torch.Tensor) shape [BxCHxHxW]
    """
    out = x
    return out 



