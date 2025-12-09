"""A module for a mixture density network layer

For more info on MDNs, see _Mixture Density Networks_ by Bishop, 1994.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical


ONEOVERSQRT2PI: float = 1.0 / math.sqrt(2 * math.pi)
LOG2PI: float = math.log(2 * math.pi)
MIN_SIGMA: float = 1e-6


class MDN(nn.Module):
    """A mixture density network layer

    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.

    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions

    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.

    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features: int, out_features: int, num_gaussians: int) -> None:
        super(MDN, self).__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.num_gaussians: int = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

    def forward(self, minibatch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        pi = self.pi(minibatch)
        sigma = torch.exp(self.sigma(minibatch))
        sigma = torch.clamp(sigma, min=MIN_SIGMA)
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


def gaussian_probability(sigma: Tensor, mu: Tensor, target: Tensor) -> Tensor:
    """Returns the probability of `target` given MoG parameters `sigma` and `mu`.

    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        target (BxI): A batch of target. B is the batch size and I is the number of
            input dimensions.

    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    target = target.unsqueeze(1).expand_as(sigma)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
    return torch.prod(ret, 2)


def log_gaussian_probability(sigma: Tensor, mu: Tensor, target: Tensor) -> Tensor:
    """Returns the log probability of `target` given MoG parameters `sigma` and `mu`.

    This is a numerically stable version that works in log-space.

    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        target (BxI): A batch of target. B is the batch size and I is the number of
            input dimensions.

    Returns:
        log_probabilities (BxG): The log probability of each point in the
            distribution in the corresponding sigma/mu index.
    """
    target = target.unsqueeze(1).expand_as(sigma)
    # Log of Gaussian PDF: -0.5 * log(2*pi) - log(sigma) - 0.5 * ((x - mu) / sigma)^2
    log_prob = -0.5 * LOG2PI - torch.log(sigma) - 0.5 * ((target - mu) / sigma) ** 2
    # Sum over output dimensions (product in probability space = sum in log space)
    return torch.sum(log_prob, dim=2)


def mdn_loss(pi: Tensor, sigma: Tensor, mu: Tensor, target: Tensor) -> Tensor:
    """Calculates the error, given the MoG parameters and the target

    The loss is the negative log likelihood of the data given the MoG
    parameters. Uses logsumexp for numerical stability.
    """
    log_pi = torch.log(pi + 1e-10)
    log_prob = log_gaussian_probability(sigma, mu, target)
    # log(sum(pi * prob)) = logsumexp(log(pi) + log(prob))
    nll = -torch.logsumexp(log_pi + log_prob, dim=1)
    return torch.mean(nll)


def sample(pi: Tensor, sigma: Tensor, mu: Tensor) -> Tensor:
    """Draw samples from a MoG.
    """
    # Choose which gaussian we'll sample from
    pis = Categorical(pi).sample().view(pi.size(0), 1, 1)
    # Choose a random sample, one randn for batch X output dims
    # Do a (output dims)X(batch size) tensor here, so the broadcast works in
    # the next step, but we have to transpose back.
    gaussian_noise = torch.randn(
        (sigma.size(2), sigma.size(0)), requires_grad=False,
        device=sigma.device, dtype=sigma.dtype)
    variance_samples = sigma.gather(1, pis).detach().squeeze()
    mean_samples = mu.detach().gather(1, pis).squeeze()
    return (gaussian_noise * variance_samples + mean_samples).transpose(0, 1)
