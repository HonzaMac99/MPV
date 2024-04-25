import numpy as np
import math
import torch
import torch.nn.functional as F
import typing
from imagefiltering import *


def x_greater_in_dir(x, dir):
    if len(dir) == 2:
        x_dir, y_dir = dir
        z_dir = 0
    else:
        x_dir, y_dir, z_dir = dir

    assert x_dir in [-1, 0, 1], f"Wrong x_dir value: {x_dir}"
    assert y_dir in [-1, 0, 1], f"Wrong y_dir value: {y_dir}"
    assert z_dir in [-1, 0, 1], f"Wrong z_dir value: {z_dir}"

    new_x = x.clone()
    new_x = torch.cat((new_x[..., x_dir:], new_x[..., :x_dir]), -1)
    new_x = torch.cat((new_x[..., y_dir:, :], new_x[..., :y_dir, :]), -2)
    new_x = torch.cat((new_x[..., z_dir:, :, :], new_x[..., :z_dir, :, :]), -3)

    mask = x > new_x
    return mask


def harris_response(x: torch.Tensor, sigma_d: float, sigma_i: float, alpha: float = 0.04) -> torch.Tensor:
    r"""Computes the Harris cornerness function.The response map is computed according the following formulation:

    .. math::
        R = det(M) - alpha \cdot trace(M)^2

    where:

    .. math::
        M = \sum_{(x,y) \in W}
        \begin{bmatrix}
            I^{2}_x & I_x I_y \\
            I_x I_y & I^{2}_y \\
        \end{bmatrix}

    and :math:`k` is an empirically determined constant
    :math:`k ∈ [ 0.04 , 0.06 ]`

    Args:
        x: torch.Tensor: 4d tensor
        sigma_d (float): sigma of Gaussian derivative
        sigma_i (float): sigma of Gaussian blur, aka integration scale
        alpha (float): constant

    Return:
        torch.Tensor: Harris response

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, H, W)`
    """
    # sigma_d = sigma_d*1.3  # correct the sigma to match reference
    G = spatial_gradient_first_order(x, sigma_d)
    Gx = G[:, :, 0]
    Gy = G[:, :, 1]

    Gx2 = gaussian_filter2d(Gx*Gx, sigma_i)
    Gy2 = gaussian_filter2d(Gy*Gy, sigma_i)
    GxGy = gaussian_filter2d(Gx*Gy, sigma_i)  # GxGy = GyGx

    # M = [[ Gx2, GxGy],
    #      [GyGx,  Gy2]]

    det_M = Gx2*Gy2 - GxGy**2
    trace_M = Gx2 + Gy2

    R = det_M - alpha * trace_M**2
    return R


def nms2d(x: torch.Tensor, th: float = 0):
    r"""Applies non maxima suppression to the feature map in 3x3 neighborhood.
    Args:
        x: torch.Tensor: 4d tensor
        th (float): threshold
    Return:
        torch.Tensor: nmsed input

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, H, W)`
    """
    # h, w = x.shape[2:]
    # for i in range(1, h-1):
    #     for j in range(1, w-1):
    #         patch = x[..., i-1:i+2, j-1:j+2]
    #         mid_val = patch[..., 1, 1].item()
    #         patch[..., 1, 1] = 0                        # omit the middle in the maximisation
    #         if mid_val > max(th, patch.max().item()):
    #             out[..., i, j] = mid_val
    #         patch[..., 1, 1] = mid_val                  # return the orig val to x

    # MUCH faster with matrix operations:
    vals = [1, 0, -1]
    dirs = [[x, y] for x in vals for y in vals if (x != 0 or y != 0)]
    shifts = [x_greater_in_dir(x, dir) for dir in dirs]
    mask = (x > th)
    for shift in shifts:
        mask *= shift  # logical AND
    out = mask * x
    return out


def harris(x: torch.Tensor, sigma_d: float, sigma_i: float, th: float = 0):
    r"""Returns the coordinates of maximum of the Harris function.
    Args:
        x: torch.Tensor: 4d tensor
        sigma_d (float): scale
        sigma_i (float): scale
        th (float): threshold

    Return:
        torch.Tensor: coordinates of local maxima in format (b,c,h,w)

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(N, 4)`, where N - total number of maxima and 4 is (b,c,h,w) coordinates
    """
    # To get coordinates of the responces, you can use torch.nonzero function
    x_harris = harris_response(x, sigma_d, sigma_i)
    x_harris_nms2d = nms2d(x_harris, th)
    out = torch.nonzero(x_harris_nms2d)
    return out


def create_scalespace(x: torch.Tensor, n_levels: int, sigma_step: float):
    r"""Creates an scale pyramid of image, usually used for local feature
    detection. Images are consequently smoothed with Gaussian blur.
    Args:
        x: torch.Tensor :math:`(B, C, H, W)`
        n_levels (int): number of the levels.
        sigma_step (float): blur step.

    Returns:
        Tuple(torch.Tensor, List(float)):
        1st output: image pyramid, (B, C, n_levels, H, W)
        2nd output: sigmas (coefficients for scale conversion)
    """

    b, ch, h, w = x.size()
    out = torch.zeros(b, ch, n_levels, h, w)
    sigmas = []

    sigma_d = 1
    sigmas.append(sigma_d)
    out[:, :, 0, :, :] = x
    for i in range(1, n_levels):
        sigma_d *= sigma_step
        sigmas.append(sigma_d)
        out[:, :, i, :, :] = gaussian_filter2d(x, sigma_d)

    return out, sigmas


def nms3d(x: torch.Tensor, th: float = 0):
    r"""Applies non maxima suppression to the scale space feature map in 3x3x3 neighborhood.
    Args:
        x: torch.Tensor: 5d tensor
        th (float): threshold
    Shape:
      - Input: :math:`(B, C, D, H, W)`
      - Output: :math:`(B, C, D, H, W)`
    """
    # d, h, w = x.shape[2:]
    # out = torch.zeros_like(x)
    # for i in range(1, d-1):
    #     for j in range(1, h-1):
    #         for k in range(1, w-1):
    #             patch = x[..., i-1:i+2, j-1:j+2, k-1:k+2]
    #             mid_val = patch[..., 1, 1, 1]
    #             patch[..., 1, 1, 1] = 0                 # omit the middle in the maximisation
    #             if mid_val > max(th, patch.max()):
    #                 out[..., i, j, k] = mid_val
    #             patch[..., 1, 1, 1] = mid_val           # return the orig val to x

    # MUCH faster with matrix operations:
    vals = [1, 0, -1]
    dirs = [[x, y, z] for x in vals for y in vals for z in vals if (x != 0 or y != 0 or z != 0)]
    shifts = [x_greater_in_dir(x, dir) for dir in dirs]
    mask = (x > th)
    for shift in shifts:
        mask *= shift  # logical AND
    out = mask * x
    return out


def scalespace_harris_response(x: torch.Tensor,
                                n_levels: int = 40,
                                sigma_step: float = 1.1):
    r"""First computes scale space and then computes the Harris cornerness function 
    Args:
        x: torch.Tensor: 4d tensor
        n_levels (int): number of the levels, (default 40)
        sigma_step (float): blur step, (default 1.1)

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, N_LEVELS, H, W)`, List(floats)
    """

    sigma_i = 2.0  # for 3x3 gaussian kernel
    ss, sigmas_d = create_scalespace(x, n_levels, sigma_step)
    out = torch.zeros_like(ss)
    for i in range(n_levels):
        x_blurred = ss[:, :, i, :, :]
        h_response = harris_response(x_blurred, sigmas_d[i], sigma_i)
        out[:, :, i, :, :] = sigmas_d[i]**2 * h_response
        print(f"Got {i}-th harris response")
    return out, sigmas_d


def scalespace_harris(x: torch.Tensor,
                       th: float = 0,
                       n_levels: int = 40,
                       sigma_step: float = 1.1):
    r"""Returns the coordinates of maximum of the Harris function.
    Args:
        x: torch.Tensor: 4d tensor
        th (float): threshold
        n_levels (int): number of scale space levels (default 40)
        sigma_step (float): blur step, (default 1.1)
        
    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(N, 5)`, where N - total number of maxima and 5 is (b,c,d,h,w) coordinates
    """
    # To get coordinates of the responces, you can use torch.nonzero function
    # Don't forget to convert scale index to scale value with use of sigma

    x_harris, sigmas = scalespace_harris_response(x, n_levels, sigma_step)
    print("got x_harris")
    x_harris_nms3d = nms3d(x_harris, th)
    print("got nms3d")
    out = torch.nonzero(x_harris_nms3d)
    print(out)
    return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import kornia
    import cv2

    # Visualization functions
    def plot_torch(x, y, *kwargs):
        plt.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy(), *kwargs)
        return


    def imshow_torch(tensor, figsize=(8, 6), *kwargs):
        plt.figure(figsize=figsize)
        plt.imshow(kornia.tensor_to_image(tensor), *kwargs)
        return


    def imshow_torch_channels(tensor, dim=1, *kwargs):
        num_ch = tensor.size(dim)
        fig = plt.figure(figsize=(num_ch * 5, 5))
        tensor_splitted = torch.split(tensor, 1, dim=dim)
        for i in range(num_ch):
            fig.add_subplot(1, num_ch, i + 1)
            plt.imshow(kornia.tensor_to_image(tensor_splitted[i].squeeze(dim)), *kwargs)
        return


    def timg_load(fname, to_gray=True):
        img = cv2.imread(fname)
        with torch.no_grad():
            timg = kornia.image_to_tensor(img, False).float()
            if to_gray:
                timg = kornia.color.bgr_to_grayscale(timg)
            else:
                timg = kornia.color.bgr_to_rgb(timg)
        return timg


    def visualize_detections(img, keypoint_locations, img_idx=0, increase_scale=1.):
        # Select keypoints relevant to image
        kpts = [cv2.KeyPoint(b_ch_sc_y_x[4].item(),
                             b_ch_sc_y_x[3].item(),
                             increase_scale * b_ch_sc_y_x[2].item())
                for b_ch_sc_y_x in keypoint_locations if b_ch_sc_y_x[0].item() == img_idx]
        vis_img = None
        vis_img = cv2.drawKeypoints(kornia.tensor_to_image(img).astype(np.uint8),
                                    kpts,
                                    vis_img,
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure(figsize=(12, 10))
        plt.imshow(vis_img)
        return


    img_corners = timg_load('corners.png')

    resp_small = harris_response(img_corners, 1.6, 2.0, 0.04)
    resp_big   = harris_response(img_corners,  7.,  9., 0.04)

    # imshow_torch_channels(torch.cat([resp_small,
    #                                  resp_big], dim=0), 0)

    # plt.show()

    # keypoint_locations = harris(img_corners, 1.6, 2.0, 0.0001)
    # print(keypoint_locations)

    # nmsed_harris = nms2d(resp_small, 1e-6)

    # imshow_torch(nmsed_harris)

    # plt.show()
    # ----------------------------------------------------------------------------


    with torch.no_grad():
        keypoint_locations = scalespace_harris(img_corners, 0.00001)

    visualize_detections(img_corners * 255., keypoint_locations, increase_scale=8.0)

    plt.show()
