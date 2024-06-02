import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from typing import Tuple
from imagefiltering import * 
from local_detector import *


# ---- Notebook stuff for debuging -------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import kornia
import kornia as K

def plot_torch(x, y, *kwargs):
    plt.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy(), *kwargs)
    return

def imshow_torch(tensor,figsize=(8,6), *kwargs):
    plt.figure(figsize=figsize)
    plt.imshow(kornia.tensor_to_image(tensor), *kwargs)
    return

def imshow_torch_channels(tensor, dim = 1, *kwargs):
    num_ch = tensor.size(dim)
    fig=plt.figure(figsize=(num_ch*5,5))
    tensor_splitted = torch.split(tensor, 1, dim=dim)
    for i in range(num_ch):
        fig.add_subplot(1, num_ch, i+1)
        plt.imshow(kornia.tensor_to_image(tensor_splitted[i].squeeze(dim)), *kwargs)
    return

def timg_load(fname, to_gray = True):
    img = cv2.imread(fname)
    with torch.no_grad():
        timg = kornia.image_to_tensor(img, False).float()
        if to_gray:
            timg = kornia.color.bgr_to_grayscale(timg)
        else:
            timg = kornia.color.bgr_to_rgb(timg)
    return timg


def visualize_detections(img, keypoint_locations, img_idx = 0, increase_scale = 1.):
    # Select keypoints relevant to image   
    kpts = [cv2.KeyPoint(b_ch_sc_y_x[4].item(),
                         b_ch_sc_y_x[3].item(),
                         increase_scale * b_ch_sc_y_x[2].item(), 90)
            for b_ch_sc_y_x in keypoint_locations if b_ch_sc_y_x[0].item() == img_idx]
    vis_img = None
    vis_img = cv2.drawKeypoints(kornia.tensor_to_image(img).astype(np.uint8),
                                kpts,
                                vis_img, 
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(12,10))
    plt.imshow(vis_img)
    return
# ----------------------------------------------------------------------------------------------------------------------


def affine_from_location(b_ch_d_y_x: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
    r"""Computes transformation matrix A which transforms point in homogeneous coordinates from canonical coordinate system into image
    from keypoint location (output of scalespace_harris or scalespace_hessian)
    Return:
        torch.Tensor: affine tranformation matrix

    Shape:
        - Input :math:`(B, 5)`
        - Output: :math:`(B, 3, 3)`, :math:`(B, 1)`

    """
    # input_format: [[batch, channel, scale, y_coord, x_coord], ... ]
    s = b_ch_d_y_x[:, 2]
    y = b_ch_d_y_x[:, 3]
    x = b_ch_d_y_x[:, 4]

    A = torch.zeros(b_ch_d_y_x.size(0), 3, 3)
    A[:, 0, 0] = s
    A[:, 1, 1] = s
    A[:, 0, 2] = x
    A[:, 1, 2] = y
    A[:, 2, 2] = 1

    img_idxs = b_ch_d_y_x[:, 0].unsqueeze(1).long()

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
    A_norm, img_idxs = affine_from_location(b_ch_d_y_x)
    c = torch.cos(ori).reshape(-1)
    s = torch.sin(ori).reshape(-1)

    R = torch.zeros(b_ch_d_y_x.size(0), 3, 3)
    R[:, 0, 0] = c
    R[:, 0, 1] = s
    R[:, 1, 0] = -s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1

    A = A_norm @ R

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
    a12_diff = aff_shape[:, 0].reshape(-1) / 2
    a11_diff = aff_shape[:, 1].reshape(-1) / 2
    a21_diff = aff_shape[:, 2].reshape(-1) / 2

    A, img_idxs = affine_from_location_and_orientation(b_ch_d_y_x, ori)
    A[:, 0, 1] -= a12_diff
    A[:, 1, 0] -= a21_diff

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

    for i in range(ang_bins_vec.shape[-1]):
        vals = len_vec[:, i]
        bins = ang_bins_vec[:, i]
        hist[:, bins] += vals  # increment hist across all batches

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
    b, c, h, w = input.shape
    margin_size = 2
    sb_offsets = int(h // num_spatial_bins)
    sb_size = sb_offsets + margin_size * 2

    # prepare a weighting gaussian window-function for every subpatch
    # todo: use the saw w func instead?
    center = int(sb_size // 2)
    gauss_mask = torch.zeros_like(input)
    gauss_mask[..., center, center] = 1
    gauss_mask = gaussian_filter2d(gauss_mask, 1)

    # compute orientation and magnitude
    G = spatial_gradient_first_order(input, 1)
    Gx = G[:, :, 0]
    Gy = G[:, :, 1]
    G_ang = torch.atan2(Gy, Gx)
    G_len = torch.sqrt(Gx**2 + Gy**2)  # no global weighting here, it is done per subpatch

    # ang_bins = torch.floor((G_ang + torch.pi)*(num_ang_bins-1) / (2*torch.pi)).long()
    # lengths = G_len

    # include the extra checks?

    hist = torch.zeros((b, num_ang_bins * num_spatial_bins**2), dtype=torch.float32)

    for sb_y in range(num_spatial_bins):
        y_offset = sb_y * sb_offsets - margin_size
        y_s, y_gs = max(y_offset, 0),             max(0, -y_offset)
        y_e, y_ge = min(y_offset + sb_size, h),   min(sb_size, h-y_offset)

        # note: computing separate indexes for gauss w-func to be aligned,
        # if the overlapping subpatch is cropped at the side of the image

        for sb_x in range(num_spatial_bins):
            x_offset = sb_x * sb_offsets - margin_size
            x_s, x_gs = max(x_offset, 0),             max(0, -x_offset)
            x_e, x_ge = min(x_offset + sb_size, w),   min(sb_size, w-x_offset)

            sb_id = (sb_y*num_spatial_bins + sb_x)
            print(f"computing patch {sb_id}")

            # get the subpatch part of the global ang and len values
            G_ang_sp = G_ang[..., y_s:y_e, x_s:x_e]
            ang_bins = torch.floor((G_ang_sp + torch.pi)*(num_ang_bins-1) / (2*torch.pi)).long()
            lengths = G_len[..., y_s:y_e, x_s:x_e] * gauss_mask[..., y_gs:y_ge, x_gs:x_ge]
            # TODO: use G_len[].clone() here?

            for y in range(y_e-y_s):
                for x in range(x_e-x_s):
                    vals = lengths[... , y, x]
                    bins = ang_bins[... , y, x] + num_ang_bins * sb_id
                    hist[:, bins] += vals  # increment hist across all batches

    return hist


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

    timg = timg_load('graffiti.ppm', False) / 255.
    timg_gray = kornia.color.rgb_to_grayscale(timg)
    imshow_torch(timg)

    with torch.no_grad():
        keypoint_locations = scalespace_harris(timg_gray, 0.00001)
        visualize_detections(timg_gray * 255., keypoint_locations, increase_scale=16.0)

    A, img_idxs = affine_from_location(keypoint_locations)

    print(f'keypoint_locations={keypoint_locations[100]}')
    print(f'A={A[100]}')

    idxs = range(100, 105)
    visualize_detections(timg_gray * 255., keypoint_locations[idxs], increase_scale=6.0)
    patches = extract_affine_patches(timg, A[idxs], img_idxs[idxs], 32, 6.0 / 2.)
    # OpenCV visualization treats keypoint scale as diameter, while extract_affine_patches as radius,
    # therefore we should divide by two to get same area

    imshow_torch_channels(patches, 0)


    # def normalize_angle(ang):
    #     # https://stackoverflow.com/a/22949941/1983544
    #     return ang - (torch.floor((ang + K.pi) / (2.0 * K.pi))) * 2.0 * K.pi


    # def benchmark_orientation_consistency(orienter, patches, PS_out, angles=[15], bins=36):
    #     import kornia as K
    #     from kornia.geometry.transform import center_crop, rotate
    #     from kornia.geometry.conversions import rad2deg, deg2rad
    #     errors = []
    #     with torch.no_grad():
    #         patches_orig_crop = center_crop(patches, (PS_out, PS_out))
    #         ang_out = normalize_angle(orienter(patches_orig_crop, bins))
    #         for ang_gt in angles:
    #             ang_gt = torch.tensor(ang_gt)
    #             patches_ang = rotate(patches, ang_gt)
    #             patches_ang_crop = center_crop(patches_ang, (PS_out, PS_out))
    #             ang_out_ang = normalize_angle(orienter(patches_ang_crop, bins))
    #             error_aug_cw = normalize_angle(ang_out - deg2rad(ang_gt) - ang_out_ang).abs()
    #             error_aug_ccw = normalize_angle(ang_out + deg2rad(ang_gt) - ang_out_ang).abs()
    #             if error_aug_ccw.mean().item() < error_aug_cw.mean().item():
    #                 error_aug = error_aug_ccw
    #             else:
    #                 error_aug = error_aug_cw
    #             errors.append(error_aug.mean())
    #             print(f'mean consistency error = {rad2deg(error_aug.mean()):.1f} [deg]')
    #     return rad2deg(torch.stack(errors).mean())


    # dir_fname = 'patches/'
    # fnames = os.listdir(dir_fname)
    # angles = [90., 60., 45., 30.]
    # PS_out = 32
    # PS = 65

    # angles = [70., 30.]
    # orienter = estimate_patch_dominant_orientation
    # with torch.no_grad():
    #     errors = []
    #     for f in fnames[::-1]:
    #         fname = os.path.join(dir_fname, f)
    #         patches = K.image_to_tensor(np.array(Image.open(fname).convert("L"))).float() / 255.
    #         patches = patches.reshape(-1, 1, PS, PS)
    #         err = benchmark_orientation_consistency(orienter, patches, PS_out, angles)
    #         errors.append(err)
    #     AVG_ERR = torch.stack(errors).mean().item()
    #     print(f'Average error = {AVG_ERR:.1f} deg')

