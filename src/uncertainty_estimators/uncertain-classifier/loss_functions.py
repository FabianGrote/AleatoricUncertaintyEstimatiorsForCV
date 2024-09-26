import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

class Loss(torch.nn.Module):
  def __init__(self, num_classes=10, T=1000):
    super(Loss,self).__init__()
    self.mvn = MultivariateNormal(torch.zeros(num_classes), torch.eye(num_classes))
    self.T = T
    self.class_count = num_classes

  def forward(self, output, y, device):
    # Equation 6 in the paper
    # => mu = softmax output or logits, sigma = observation noise parameter that captures noise in the output/ variance?
    mu, log_sigma2 = output
    y_hat = []
    
    # T is the number of sampled masked model weights
    for t in range(self.T):
      epsilon = self.mvn.sample((len(mu),)).to(device)
      # log_softmax allows to retrieve log-probabilities
      # upper part of equation 12 in paper. But wonder where 0.5 comes from
      numerator = F.log_softmax(mu + torch.exp(0.5*log_sigma2)*epsilon, dim=1)
      y_hat.append( numerator - np.log(self.T))

    y_hat = torch.stack(tuple(y_hat))
    # first param = input, second param is target
    return F.nll_loss(torch.logsumexp(y_hat, dim=0), y)
  

# Bayesian categorical cross entropy.
  # N data points, C classes, T monte carlo simulations
  # true - true values. Shape: (N, C)
  # pred_var - predicted logit values and variance. Shape: (N, C + 1)
  # returns - loss (N,)
class BayesianCategoricalCrossEntropy_KylesVersion(torch.nn.Module):
  def __init__(self, T=100, num_classes=10):
    self.T = T
    self.num_classes = num_classes

  def forward(self, true, pred_var):
    # shape: (N,)
    std = torch.sqrt(pred_var[:, self.num_classes:])
    # shape: (N,)
    variance = pred_var[:, self.num_classes]
    variance_depressor = torch.ex(variance) - torch.ones_like(variance)
    # shape: (N, C)
    pred = pred_var[:, 0:self.num_classes]
      
    # shape: (N,)
    undistorted_loss = nn.CrossEntropyLoss(pred, true)

    iterable = torch.ones(self.T)
    dist = torch.normal(mean=torch.zeros_like(std), scale=std)
    # monte carlo
    monte_carlo_results = torch.vmap(self.gaussian_categorical_crossentropy(true, pred, dist, undistorted_loss, self.num_classes))(iterable)
    variance_loss = torch.mean(monte_carlo_results, axis=0) * undistorted_loss
      
    return variance_loss + undistorted_loss + variance_depressor

  # for a single monte carlo simulation, 
  #   calculate categorical_crossentropy of 
  #   predicted logit values plus gaussian 
  #   noise vs true values.
  # true - true values. Shape: (N, C)
  # pred - predicted logit values. Shape: (N, C)
  # dist - normal distribution to sample from. Shape: (N, C)
  # undistorted_loss - the crossentropy loss without variance distortion. Shape: (N,)
  # num_classes - the number of classes. C
  # returns - total differences for all classes (N,)
  def gaussian_categorical_crossentropy(self, true, pred, dist, undistorted_loss, num_classes):
    def map_fn(i):
      std_samples = torch.transpose(dist.sample(num_classes))
      distorted_loss = nn.CrossEntropyLoss(pred + std_samples, true)
      diff = undistorted_loss - distorted_loss
      return -nn.ELU()(diff)
    return map_fn


# Bayesian categorical cross entropy.
# N data points, C classes, T monte carlo simulations
# true - true values. Shape: (N, C)
# pred_var - predicted logit values and variance. Shape: (N, C + 1)
# returns - loss (N)
class BayesianCategoricalCrossEntropy_KendallAndGalsVersion(torch.nn.Module):
  def __init__(self, T=100, num_classes=10):
    self.T = T
    self.num_classes = num_classes

  def forward(self, true, pred_var):
    # shape: [N, 1]
    std_vals = torch.sqrt(pred_var[:, self.num_classes:])
    # shape: [N, C]
    std = true * std_vals
    pred = pred_var[:, 0:self.num_classes]
    iterable = torch.ones(self.T)
    dist = torch.normal(mean=torch.zeros_like(std), scale=std)
    # Shape: (T, N)
    monte_carlo_results = torch.vmap(self.gaussian_categorical_crossentropy(true, pred, dist))(iterable)
    return torch.mean(monte_carlo_results, axis=0)

  # for a single monte carlo simulation, 
  #   calculate categorical_crossentropy of 
  #   predicted logit values plus gaussian 
  #   noise vs true values.
  # true - true values. Shape: (N, C)
  # pred - predicted logit values. Shape: (N, C)
  # dist - normal distribution to sample from. Shape: (N, C)
  # returns - categorical_crossentropy for each sample (N)
  def gaussian_categorical_crossentropy(self, true, pred, dist):
    def map_fn(i):
      return nn.CrossEntropyLoss(pred + dist.sample(1), true)
    return map_fn
  