import numpy as np
import math
import torch
import torch.nn.functional as F
import typing


def hdist(H: torch.Tensor, pts_matches: torch.Tensor):
    '''Function, calculates one-way reprojection error
    
    Return:
        torch.Tensor: per-correspondence Eucledian squared error


    Shape:
        - Input :math:`(3, 3)`, :math:`(B, 4)`
        - Output: :math:`(B, 1)`
    '''

    N = pts_matches.shape[0]
    ones_arr = torch.ones(1, N)

    pts_1 = pts_matches[:, :2].T
    pts_2 = pts_matches[:, 2:].T

    pts_1_proj = H @ torch.concat((pts_1, ones_arr))
    pts_1_proj = pts_1_proj[:2, :] / pts_1_proj[2, :]

    dists = ((pts_2 - pts_1_proj)**2).sum(0)
    return dists.unsqueeze(1)


def sample(pts_matches: torch.Tensor, num: int = 4):
    '''Function, which draws random sample from pts_matches
    
    Return:
        torch.Tensor:

    Args:
        pts_matches: torch.Tensor: 2d tensor
        num (int): number of correspondences to sample

    Shape:
        - Input :math:`(B, 4)`
        - Output: :math:`(num, 4)`
    '''
    # torch.random_sample() is also a possibility

    rng = np.random.default_rng()
    sample = rng.choice(pts_matches, size=num, replace=False)

    return torch.tensor(sample)


def orient_pred(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def getH(min_sample):
    '''Function, which estimates homography from minimal sample
    Return:
        torch.Tensor:

    Args:
        min_sample: torch.Tensor: 2d tensor

    Shape:
        - Input :math:`(B, 4)`
        - Output: :math:`(3, 3)`
    '''

    # check for collinear triplets (degenerate case)
    th = 0.05   # in Brute at val 0.0329 we get collinearity
    for i in range(4):
        p1, p_1 = min_sample[      i, :2], min_sample[      i, 2:4]
        p2, p_2 = min_sample[(i+1)%4, :2], min_sample[(i+1)%4, 2:4]
        p3, p_3 = min_sample[(i+2)%4, :2], min_sample[(i+2)%4, 2:4]
        if abs(orient_pred(p1, p2, p3)) < th:
            # print("Warn: degeneracy encountered in img1: ", p1, p2, p3)
            return None
        if abs(orient_pred(p_1, p_2, p_3)) < th:
            # print("Warn: degeneracy encountered in img2: ", p_1, p_2, p_3)
            return None

    # compute the homography
    C = torch.zeros(8, 9)
    for i in range(4):
        p, q = min_sample[i, :2], min_sample[i, 2:]
        C[i*2, :]   = torch.tensor([-p[0], -p[1], -1,      0,      0,  0, q[0]*p[0], q[0]*p[1], q[0]])
        C[i*2+1, :] = torch.tensor([     0,      0,  0, -p[0], -p[1], -1, q[1]*p[0], q[1]*p[1], q[1]])

    U, S, Vt = torch.linalg.svd(C)
    h = Vt[-1, :]  # always the last row
    h_norm = h / h[-1]

    H_norm = h_norm.reshape(3, 3)
    return H_norm


# num_tc = number of tentative correspondences
def nsamples(n_inl:int , num_tc:int , sample_size:int , conf: float):
    inl_ratio = (n_inl+1) / num_tc   # adding 1 to avoid division by 0 = np.log(1)
    if inl_ratio >= 1:
        return 0
    else:
        return np.log(1 - conf) / np.log(1 - inl_ratio**sample_size)


def ransac_h(pts_matches: torch.Tensor, th: float = 4.0, conf: float = 0.95, max_iter:int = 10000):
    '''Function, which robustly estimates homography from noisy correspondences
    
    Return:
        torch.Tensor: per-correspondence Eucledian squared error

    Args:
        pts_matches: torch.Tensor: 2d tensor
        th (float): pixel threshold for correspondence to be counted as inlier
        conf (float): confidence
        max_iter (int): maximum iteration, overrides confidence
        
    Shape:
        - Input  :math:`(B, 4)`
        - Output: :math:`(3, 3)`,   :math:`(B, 1)`
    '''
    n_corresp = pts_matches.shape[0]
    sample_size = 4

    n_iter = 0
    max_inliers = 0
    H_best = torch.eye(3)
    inls_best = np.array([])
    while (n_iter < max_iter):
        min_sample = sample(pts_matches, sample_size)
        H = getH(min_sample)
        if H is None:
            continue
        dists = hdist(H, pts_matches)
        inls = (dists < th)
        n_inliers = inls.flatten().sum().item()

        if n_inliers > max_inliers:
            print(f"Got new best support {n_inliers} at iteration {n_iter}")
            max_inliers = n_inliers
            H_best = H
            inls_best = inls

        if n_inliers == n_corresp:
            break

        max_iter = nsamples(n_inliers, n_corresp, sample_size, conf)
        n_iter += 1

    return H_best, inls_best

