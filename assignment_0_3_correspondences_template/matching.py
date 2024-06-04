import numpy as np
import math
import torch
import torch.nn.functional as F
import typing




def match_snn(desc1: torch.Tensor, desc2: torch.Tensor, th: float = 0.8):
    '''Function, which finds nearest neightbors for each vector in desc1,
    which satisfy first to second nearest neighbor distance <= th check
    
    Return:
        torch.Tensor: indexes of matching descriptors in desc1 and desc2
        torch.Tensor: L2 desriptor distance ratio 1st to 2nd nearest neighbor


    Shape:
        - Input :math:`(B1, D)`, :math:`(B2, D)`
        - Output: :math:`(B3, 2)`, :math:`(B3, 1)` where 0 <= B3 <= B1
    '''
    d_matrix = torch.cdist(desc1, desc2)

    matches_idxs = []
    match_dists = []
    for i in range(d_matrix.shape[0]):
        nearest_idxs = torch.argsort(d_matrix[i])
        nn, snn = nearest_idxs[:2].tolist()
        nn_d, snn_d = d_matrix[i, [nn, snn]].tolist()
        if nn_d / snn_d < th:
            matches_idxs.append([i, nn])
            match_dists.append(nn_d / snn_d)

    matches_idxs = torch.tensor(matches_idxs)
    match_dists = torch.tensor(match_dists)

    return matches_idxs, match_dists
