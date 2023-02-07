import torch
from pykeops.torch import LazyTensor

class GaussianKDE(torch.nn.Module):

    def __init__(self, data, weights=None, sigma=3):
        r"""
        Inputs:
            :data: Tensor (N_points, N_dims)
            :weights: Tensor (N_points)
            :sigma: Gaussian scale
        """
        super().__init__()
        self.dpoints = data
        if weights is not None:
            self.weights = weights.unsqueeze(1)
        else:
            self.weights = None
        self.sigma = sigma

    def forward(self, x):
        r"""
        Apply the kde at the given locations.

        Inputs:
            :x: Tensor (B, N_locs, N_dims)
        """
        a = LazyTensor(x.unsqueeze(0))
        b = LazyTensor(self.dpoints.unsqueeze(1).unsqueeze(0).contiguous())
        pairwise_ = (- (a-b).square().sum(-1) / self.sigma).exp() # (N_points, N_locs)
        if self.weights is not None:
            w_ = LazyTensor(self.weights.unsqueeze(0).unsqueeze(-1))
            outp_ = (pairwise_ * w_).sum(0,1)
        else:
            outp_ = pairwise_.sum(0,1)

        return outp_

class ParabolicKDE(torch.nn.Module):

    def __init__(self, data, weights=None, sigma=3):
        r"""
        Inputs:
            :data: Tensor (N_points, N_dims)
            :weights: Tensor (N_points)
            :sigma: scale
        """
        super().__init__()
        self.dpoints = data
        if weights is not None:
            self.weights = weights.unsqueeze(1)
        else:
            self.weights = None
        self.sigma = sigma

    def forward(self, x):
        r"""
        Apply the kde at the given locations.

        Inputs:
            :x: Tensor (B, N_locs, N_dims)
        """
        a = LazyTensor(x.unsqueeze(0))
        b = LazyTensor(self.dpoints.unsqueeze(1).unsqueeze(0).contiguous())
        pairwise_ = ((a-b)/self.sigma).square().sum(-1) # (N_points, N_locs)
        if self.weights is not None:
            w_ = LazyTensor(self.weights.unsqueeze(0).unsqueeze(-1))
            outp_ = (pairwise_ * w_).sum(0,1)
        else:
            outp_ = pairwise_.sum(0,1)

        return outp_
