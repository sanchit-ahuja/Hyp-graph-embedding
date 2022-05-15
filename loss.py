import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

def focal_loss(labels, logits, alpha, gamma):
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma, device):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)


    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=-1)
        samples_per_cls = torch.Tensor(samples_per_cls).to(device)
        cb_loss = F.cross_entropy(input=pred, target=labels_one_hot.argmax(dim=-1))


    return cb_loss

def loss_fn(output, targets, n_classes, samples_per_cls, device, beta, gamma):
    beta = beta
    gamma = gamma
    no_of_classes = n_classes
    loss_type = "softmax"

    return CB_loss(targets, output, samples_per_cls, no_of_classes, loss_type, beta, gamma, device)