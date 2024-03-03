import numpy as np
import math
import torch
import torch.nn.functional as F
import typing


def get_gausskernel_size(sigma, force_odd = True):
    ksize = 2 * math.ceil(sigma * 3.0) + 1
    if ksize % 2 == 0 and force_odd:
        ksize += 1
    return int(ksize)


def gaussian1d(x: torch.Tensor, sigma: float) -> torch.Tensor: 
    '''Function that computes values of a (1D) Gaussian with zero mean and variance sigma^2'''
    a = 1/(sigma*np.sqrt(2*np.pi))
    out =  a * torch.exp(-torch.pow(x, 2)/(2*np.power(sigma, 2)))
    return out


def gaussian_deriv1d(x: torch.Tensor, sigma: float) -> torch.Tensor:  
    '''Function that computes values of a (1D) Gaussian derivative'''
    a = -1 / (np.power(sigma, 3)*np.sqrt(2*np.pi))
    out =  a * x * torch.exp(-torch.pow(x, 2)/(2*np.power(sigma, 2)))
    return out


def filter2d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Function that convolves a tensor with a kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(kH, kW)`.
    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.
    """
    weights = torch.zeros(1, 1, kernel.shape[0], kernel.shape[1])
    weights[0, 0, ...] = kernel

    ## Do not forget about flipping the kernel!
    ## See in details here https://towardsdatascience.com/convolution-vs-correlation-af868b6b4fb5
    weights_flipped = torch.flip(weights, dims=[2, 3])

    x_out = F.conv2d(x, weights_flipped, padding=1)
    return x_out

def gaussian_filter2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r"""Function that blurs a tensor using a Gaussian filter.

    Arguments:
        sigma (Tuple[float, float]): the standard deviation of the kernel.
        
    Returns:
        Tensor: the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    """ 
    ksize = get_gausskernel_size(sigma)
    pad_size = ksize/2 if ksize % 2 == 0 else int((ksize-1)/2)
    kernel = torch.zeros(1, 1, ksize, ksize)
    for i in range(ksize):
        i_dif = i - pad_size  # y_coords
        for j in range(ksize):
            j_dif = j - pad_size  # x_coords

            a = 1 / (2 * np.pi * np.power(sigma, 2))
            b = -(np.power(i_dif, 2) + np.power(j_dif, 2)) / (2 * np.power(sigma, 2))

            kernel[..., i, j] = a * np.exp(b)

    kernel_flipped = torch.flip(kernel, dims=[2, 3])

    x_out = torch.zeros(x.shape)
    for i in range(x.shape[1]):
        x_out[:, i, :, :] = F.conv2d(x[:, i, :, :], kernel_flipped, padding=pad_size)
    return x_out


def spatial_gradient_first_order(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r"""Computes the first order image derivative in both x and y directions using Gaussian derivative

    Return:
        torch.Tensor: spatial gradients

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 2, H, W)`

    """
    bs, cs, h, w = x.shape
    ksize = get_gausskernel_size(sigma)
    pad_size = ksize/2 if ksize % 2 == 0 else int((ksize-1)/2)

    kernel = torch.zeros(1, 1, 2, ksize, ksize)
    for i in range(ksize):
        i_dif = i - pad_size  # y_coords
        for j in range(ksize):
            j_dif = j - pad_size  # x_coords

            a1 = - 1 / (np.power(sigma, 4) * 2 * np.pi) * j_dif  # x direction
            a2 = - 1 / (np.power(sigma, 4) * 2 * np.pi) * i_dif  # y direction

            b = -(np.power(i_dif, 2) + np.power(j_dif, 2)) / (2 * np.power(sigma, 2))

            kernel[..., 0, i, j] = a1 * np.exp(b)
            kernel[..., 1, i, j] = a2 * np.exp(b)

    kernel_flipped = torch.flip(kernel, dims=[3, 4])

    x_out = torch.zeros(bs, cs, 2, h, w)
    for i in range(cs):
        x_out[:, i, 0, ...] = F.conv2d(x[:, i, ...], kernel_flipped[:, :, 0, ...], padding=pad_size)
        x_out[:, i, 1, ...] = F.conv2d(x[:, i, ...], kernel_flipped[:, :, 1, ...], padding=pad_size)
    return x_out


def affine(center: torch.Tensor, unitx: torch.Tensor, unity: torch.Tensor) -> torch.Tensor:
    r"""Computes transformation matrix A which transforms point in homogeneous coordinates from canonical coordinate system into image

    Return:
        torch.Tensor: affine tranformation matrix

    Shape:
        - Input :math:`(B, 2)`, :math:`(B, 2)`, :math:`(B, 2)` 
        - Output: :math:`(B, 3, 3)`

    """
    assert center.size(0) == unitx.size(0)
    assert center.size(0) == unity.size(0)
    B = center.size(0)

    b = torch.vstack((center[:, 0],
                      center[:, 1],
                      unitx[:, 0],
                      unitx[:, 1],
                      unity[:, 0],
                      unity[:, 1]))

    # print(b)
    # print("-----------------")
    # print(b.reshape(6, B))

    A = np.zeros((6, 6))
    u = [0, 0, 1, 0, 0, 1]
    for i in range(3):
        A[i*2, :]   = np.array([u[i*2], u[i*2+1], 1, 0, 0, 0])
        A[i*2+1, :] = np.array([0, 0, 0, u[i*2], u[i*2+1], 1])

    # solve Ax = b
    x = np.linalg.solve(A, b)
    A = torch.zeros(B, 3, 3)
    for i in range(B):
        Ai = np.vstack((x[:3, i].T, x[3:, i].T, np.array([0, 0, 1])))
        A[i, :, :] = torch.tensor(Ai)

    if B == 1:
        return A[0, :, :]
    return A


def extract_affine_patches(input: torch.Tensor,
                           A: torch.Tensor,
                           img_idxs: torch.Tensor,
                           PS: int = 32,
                           ext: float = 6.0):
    """Extract patches defined by affine transformations A from image tensor X.
    
    Args:
        input: (torch.Tensor) images, :math:`(B, CH, H, W)`
        A: (torch.Tensor). :math:`(N, 3, 3)`
        img_idxs: (torch.Tensor). :math:`(N, 1)` indexes of image in batch, where patch belongs to
        PS: (int) output patch size in pixels, default = 32
        ext (float): output patch size in unit vectors. 

    Returns:
        patches: (torch.Tensor) :math:`(N, CH, PS,PS)`
    """
    b,ch,h,w = input.size()
    print("input size: ", b, ch, h, w, A.size(0))
    num_patches = A.size(0)
    # Functions, which might be useful: torch.meshgrid, torch.nn.functional.grid_sample
    # You are not allowed to use function torch.nn.functional.affine_grid
    # Note, that F.grid_sample expects coordinates in a range from -1 to 1
    # where (-1, -1) - topleft, (1,1) - bottomright and (0,0) center of the image

    patches_out = torch.zeros(num_patches, ch, PS, PS)
    for i in range(PS):  # y coords
        for j in range(PS):  # x coords
            for k in range(num_patches):
                j_dif = j*2/(PS-1) - 1  # we want range (-1, 1)
                i_dif = i*2/(PS-1) - 1

                inp_c = (A[k] @ np.array([j_dif, i_dif, 1])).type(torch.int64)
                patches_out[k, :, i, j] = input[0, :, inp_c[1], inp_c[0]]
                # patches_out[k, :, i, j] = input[0, :, i, j]

    return patches_out


def extract_antializased_affine_patches(input: torch.Tensor,
                           A: torch.Tensor,
                           img_idxs: torch.Tensor,
                           PS: int = 32,
                           ext: float = 6.0):
    """Extract patches defined by affine transformations A from scale pyramid created image tensor X.
    It runs your implementation of the `extract_affine_patches` function, so it would not work w/o it.
    You do not need to ever modify this finction, implement `extract_affine_patches` instead.
    
    Args:
        input: (torch.Tensor) images, :math:`(B, CH, H, W)`
        A: (torch.Tensor). :math:`(N, 3, 3)`
        img_idxs: (torch.Tensor). :math:`(N, 1)` indexes of image in batch, where patch belongs to
        PS: (int) output patch size in pixels, default = 32
        ext (float): output patch size in unit vectors. 

    Returns:
        patches: (torch.Tensor) :math:`(N, CH, PS,PS)`
    """
    import kornia
    b,ch,h,w = input.size()
    num_patches = A.size(0)
    scale = (kornia.feature.get_laf_scale(ext * A.unsqueeze(0)[:,:,:2,:]) / float(PS))[0]
    half: float = 0.5
    pyr_idx = (scale.log2()).relu().long()
    cur_img = input
    cur_pyr_level = 0
    out = torch.zeros(num_patches, ch, PS, PS).to(device=A.device, dtype=A.dtype)
    while min(cur_img.size(2), cur_img.size(3)) >= PS:
        _, ch_cur, h_cur, w_cur = cur_img.size()
        scale_mask = (pyr_idx == cur_pyr_level).squeeze()
        if (scale_mask.float().sum()) >= 0:
            scale_mask = (scale_mask > 0).view(-1)
            current_A = A[scale_mask]
            current_A[:, :2, :3] *= (float(h_cur)/float(h))
            patches = extract_affine_patches(cur_img,
                                 current_A, 
                                 img_idxs[scale_mask],
                                 PS, ext)
            out.masked_scatter_(scale_mask.view(-1, 1, 1, 1), patches)
        cur_img = kornia.geometry.pyrdown(cur_img)
        cur_pyr_level += 1
    return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import kornia

    def plot_torch(x, y, *kwargs):
        plt.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy(), *kwargs)
        plt.show()
        return


    def imshow_torch(tensor, *kwargs):
        plt.figure()
        plt.imshow(kornia.tensor_to_image(tensor), *kwargs)
        plt.show()
        return


    inp = torch.zeros((1, 1, 32, 32))
    inp[..., 15, 15] = 1.
    imshow_torch(inp)

    sigma = 3.0
    out = gaussian_filter2d(inp, sigma)
    imshow_torch(out)



