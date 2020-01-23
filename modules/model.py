import torch
import torch.nn as nn
import torch.nn.functional as F


class Tripletnet(nn.Module):
	def __init__(self, embeddingnet):
		super(Tripletnet, self).__init__()
		self.embeddingnet = embeddingnet

	def forward(self, x):
		embedded_x = self.embeddingnet(x)
		embedded_x = F.normalize(embedded_x, p=2, dim=1)
		dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
		dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
		return embedded_x