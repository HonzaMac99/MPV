import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from typing import Tuple
from imagefiltering import * 
from local_detector import *

import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import kornia
import kornia as K


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
    num_batches = x.shape[0]

    # create a weighting gaussian window-function
    center = int(np.floor(x.size(dim=-1)/2))
    gauss_mask = torch.zeros_like(x)
    gauss_mask[..., center, center] = 1
    gauss_mask = gaussian_filter2d(gauss_mask, 1)

    # compute orientation and magnitude
    G = spatial_gradient_first_order(x, 1)
    Gx = G[:, :, 0]
    Gy = G[:, :, 1]
    G_ang = torch.atan2(Gy, Gx)
    G_len = torch.sqrt(Gx**2 + Gy**2) * gauss_mask    # [B, 1, ps, ps] * [B, 1, ps, ps]

    ang_bins_vec = torch.floor((G_ang.flatten(-3) + torch.pi)*(num_angular_bins-1) / (2*torch.pi)).type(torch.long)
    len_vec = G_len.flatten(-3)

    # extra check, but should be in the range and not none
    ang_bins_vec[torch.isnan(ang_bins_vec)] = 0
    len_vec[torch.isnan(len_vec)] = 0
    ang_bins_vec[ang_bins_vec < 0] = 0
    ang_bins_vec[ang_bins_vec > (num_angular_bins-1)] = num_angular_bins-1

    # voting implementation
    hist = torch.zeros((num_batches, num_angular_bins), dtype=torch.float32)

    # create hist across all batches
    for i in range(ang_bins_vec.shape[-1]):
        vals = len_vec[:, i]
        bins = ang_bins_vec[:, i]
        hist[:, bins] += vals

    # perform the smoothing twice
    kernel = torch.tensor([[[1, 4, 6, 4, 1]]]) / 16                                     # kernel: [1, 1, W]
    hist = F.conv1d(F.pad(hist.unsqueeze(1), pad=(2, 2), mode="circular"), kernel)      # histog: [1, B, W]
    hist = F.conv1d(F.pad(hist, pad=(2, 2), mode="circular"), kernel)
    hist.squeeze(-2)

    max_ang_bin = torch.argmax(hist, dim=-1)
    max_ang = ((max_ang_bin / (num_angular_bins-1)) * 2 * torch.pi) - torch.pi

    return max_ang


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


if __name__ == "__main__":

    def normalize_angle(ang):
        # https://stackoverflow.com/a/22949941/1983544
        return ang - (torch.floor((ang + K.pi) / (2.0 * K.pi))) * 2.0 * K.pi


    def benchmark_orientation_consistency(orienter, patches, PS_out, angles=[15], bins=36):
        import kornia as K
        from kornia.geometry.transform import center_crop, rotate
        from kornia.geometry.conversions import rad2deg, deg2rad
        errors = []
        with torch.no_grad():
            patches_orig_crop = center_crop(patches, (PS_out, PS_out))
            ang_out = normalize_angle(orienter(patches_orig_crop, bins))
            for ang_gt in angles:
                ang_gt = torch.tensor(ang_gt)
                patches_ang = rotate(patches, ang_gt)
                patches_ang_crop = center_crop(patches_ang, (PS_out, PS_out))
                ang_out_ang = normalize_angle(orienter(patches_ang_crop, bins))
                error_aug_cw = normalize_angle(ang_out - deg2rad(ang_gt) - ang_out_ang).abs()
                error_aug_ccw = normalize_angle(ang_out + deg2rad(ang_gt) - ang_out_ang).abs()
                if error_aug_ccw.mean().item() < error_aug_cw.mean().item():
                    error_aug = error_aug_ccw
                else:
                    error_aug = error_aug_cw
                errors.append(error_aug.mean())
                print(f'mean consistency error = {rad2deg(error_aug.mean()):.1f} [deg]')
        return rad2deg(torch.stack(errors).mean())


    dir_fname = 'patches/'
    fnames = os.listdir(dir_fname)
    angles = [90., 60., 45., 30.]
    PS_out = 32
    PS = 65

    angles = [70., 30.]
    orienter = estimate_patch_dominant_orientation
    with torch.no_grad():
        errors = []
        for f in fnames[::-1]:
            fname = os.path.join(dir_fname, f)
            patches = K.image_to_tensor(np.array(Image.open(fname).convert("L"))).float() / 255.
            patches = patches.reshape(-1, 1, PS, PS)
            err = benchmark_orientation_consistency(orienter, patches, PS_out, angles)
            errors.append(err)
        AVG_ERR = torch.stack(errors).mean().item()
        print(f'Average error = {AVG_ERR:.1f} deg')

