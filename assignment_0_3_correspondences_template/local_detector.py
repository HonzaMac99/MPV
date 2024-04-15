import numpy as np
import math
import torch
import torch.nn.functional as F
import typing
from imagefiltering import *


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

    # det_M = gaussian_filter2d((Gx**2)*(Gy**2) - (Gx*Gy)**2, sigma_i)
    # trace_M = gaussian_filter2d(Gx*Gx + Gy*Gy, sigma_i)

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
    out = torch.zeros_like(x)
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
    x_detect = x_harris > th
    out = torch.nonzero(x_detect)
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
    out = torch.zeros(b, ch, n_levels, h, w), [1.0 for x in range(n_levels)]
    return out


def nms3d(x: torch.Tensor, th: float = 0):
    r"""Applies non maxima suppression to the scale space feature map in 3x3x3 neighborhood.
    Args:
        x: torch.Tensor: 5d tensor
        th (float): threshold
    Shape:
      - Input: :math:`(B, C, D, H, W)`
      - Output: :math:`(B, C, D, H, W)`
    """
    out = torch.zeros_like(x)
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
    out = torch.zeros_like(x)
    return out



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
    out = torch.zeros(0,3)
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

    img_corners = timg_load('corners.png')

    resp_small = harris_response(img_corners, 1.6, 2.0, 0.04)
    resp_big = harris_response(img_corners, 7., 9., 0.04)

    imshow_torch_channels(torch.cat([resp_small,
                                     resp_big], dim=0), 0)
    plt.show()

    keypoint_locations = harris(img_corners, 1.6, 2.0, 0.0001)
    print(keypoint_locations)
