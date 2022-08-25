import torch
import math
sigma = torch.rand(3,3,3,3)
mu = torch.rand(3,3,3,3)
target = torch.rand(3,1)

def log_gaussian_probability(sigma, mu, target, mask=None):
    """
    prob -- [B, src_len, num_gaussians]
    """
    target = target.unsqueeze(2).expand_as(sigma)
    prob = torch.log((1.0 / (math.sqrt(2 * math.pi) * sigma))) - 0.5 * ((target - mu) / sigma) ** 2
    if mask is not None:
        prob = prob.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0)
    prob = torch.sum(prob, dim=3)
    return prob

def gaussian_Exponential_family_probability(sigma, mu, target, mask=None):
    target = target.unsqueeze(2).expand_as(sigma)
    eta = mu ** 2 / sigma - 1 / 2 * (sigma ** 2)
    A_eta = (mu ** 2) / 2 * (sigma ** 2) + 1 / 2 * torch.log(2 * math.pi * sigma * sigma)
    temp = torch.matmul(eta,target)
    prob = torch.sum(torch.exp(temp-A_eta), dim=3)
    if mask is not None:
        prob = prob.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0)
    return prob
porb1 = log_gaussian_probability(sigma=sigma,mu=mu,target=target,mask=None)
prob2 = gaussian_Exponential_family_probability(sigma=sigma,mu=mu,target=target,mask=None)