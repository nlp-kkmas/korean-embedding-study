import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


class SoftmaxLayer(nn.Module):
  """ Naive softmax-layer """
  def __init__(self, output_dim, n_class):
    """

    :param output_dim: int
    :param n_class: int
    """
    super(SoftmaxLayer, self).__init__()
    self.hidden2tag = nn.Linear(output_dim, n_class)
    self.criterion = nn.CrossEntropyLoss(size_average=False)

  def forward(self, x, y):
    """

    :param x: torch.Tensor
    :param y: torch.Tensor
    :return:
    """
    tag_scores = self.hidden2tag(x)
    return self.criterion(tag_scores, y)
