# Third party imports
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage.color import deltaE_ciede2000
from scipy.spatial.distance import pdist, squareform, cdist

import torch
from torch.utils import data
import numpy as np

# Local imports
import config
from . import util
from . import data_loader


class Train():
	def __init__(self, train_images, val_images, positive_image_dict, model, criterion, optimizer, epoch):
		self.train_images = train_images
		self.val_images = val_images
		self.positive_image_dict = positive_image_dict
		self.args = config.ARGS
		self.model = model
		self.criterion = criterion							# loss function
		self.optimizer = optimizer
		self.epoch = epoch

		# Initializing the dataloader for generating images
		self.train_set = data_loader.TrainDataset(self.train_images)
		self.train_loader = data.DataLoader(self.train_set, **config.PARAMS)

		self.val_set = data_loader.TrainDataset(self.val_images)
		self.val_loader = data.DataLoader(self.val_set, **config.PARAMS)




	def _get_batch_images(self, images, positive_image_dict, transforms):
		"""generating batch images with one positive for each image from the positive image dictionary
		,so that for online training we know for sure that for every anchor image there will be one positive image
		in the batch.
		Args:
			images: list of images generated from dataloader
			positive_image_dict: dictionary having images along with their positive images
			transforms: the transformation information
		Returns:
			images for training in a batch
		"""

		batch_images = []
		labels = []
		embs = []
		for idx, image_name in enumerate(images):
			# image = image.item()
			image = Image.open(self.args.image_path + image_name)
			batch_images.append(transforms(image))
			labels.append(idx)

			# adding positive images for each of the images in the batch
			pos = random.sample(positive_image_dict[image_name],1)[0]
			image = Image.open(self.args.image_path + pos)
			batch_images.append(transforms(image))
			labels.append(idx)

		# converting the variables to torch tensor for training
		# shape (2 * batch_size)
		batch_images = torch.stack(batch_images).to(config.DEVICE)
		labels = torch.tensor(np.array(labels)).to(config.DEVICE)

		return (batch_images, labels)




	def _pairwise_distances(self, embeddings, squared=False):
		"""Compute the 2D matrix of distances between all the embeddings.
		Args:
			embeddings: tensor of shape (batch_size, embed_dim)
			squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
					 If false, output is the pairwise euclidean distance matrix.
		Returns:
			pairwise_distances: tensor of shape (batch_size, batch_size)
		"""
		

		# Get the dot product between all embeddings
		# shape (batch_size, batch_size)
		dot_product = torch.matmul(embeddings, torch.transpose(embeddings, 0, 1))

		# Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
		# This also provides more numerical stability (the diagonal of the result will be exactly 0).
		# shape (batch_size,)
		square_norm = torch.diag(dot_product)

		# Compute the pairwise distance matrix as we have:
		# ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
		# shape (batch_size, batch_size)
		distances = square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)		# here two different values for unsqueeze used to get two dimensional output

		 # Because of computation errors, some distances might be negative so we put everything >= 0.0
		distances = torch.clamp(distances, min=0.0)

		if not squared:
			# Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
			# we need to add a small epsilon where distances == 0.0
			mask = distances.eq(0.0).float()
			# mask = (torch.equal(distances, 0.0)).astype(float)
			distances = distances + mask * 1e-16

			distances = torch.sqrt(distances)

			# Correct the epsilon added: set the distances on the mask to be exactly 0.0
			distances = distances * (1.0 - mask)

		return distances




	def _get_triplet_mask(self, labels):
		"""Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
		A triplet (i, j, k) is valid if:
			- i, j, k are distinct
			- labels[i] == labels[j] and labels[i] != labels[k]
		Args:
			labels:  `Tensor` with shape [batch_size]
		"""
		# Check that i, j and k are distinct
		indices_equal = torch.eye(len(labels)).type(torch.ByteTensor).to(config.DEVICE)
		indices_not_equal = ~(indices_equal)
		i_not_equal_j = indices_not_equal.unsqueeze(2)
		i_not_equal_k = indices_not_equal.unsqueeze(1)
		j_not_equal_k = indices_not_equal.unsqueeze(0)

		distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k


		# Check if labels[i] == labels[j] and labels[i] != labels[k]
		label_equal = labels.unsqueeze(0).eq(labels.unsqueeze(1))

		i_equal_j = label_equal.unsqueeze(2)
		i_equal_k = label_equal.unsqueeze(1)

		valid_labels = i_equal_j & ~i_equal_k

		# Combine the two masks
		mask = distinct_indices & valid_labels

		return mask




	def _get_anchor_positive_triplet_mask(self, labels):
		"""Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
		Args:
			labels: torch tensor with shape [batch_size]
		Returns:
			mask: torch tensor with shape [batch_size, batch_size], boolean output
		"""
		# Check that i and j are distinct
		indices_equal = torch.eye(len(labels)).type(torch.ByteTensor).to(config.DEVICE)
		indices_not_equal = ~(indices_equal)

		# Check if labels[i] == labels[j]
		# Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
		labels_equal = labels.unsqueeze(0).eq(labels.unsqueeze(1))

		# Combine the two masks
		mask = indices_not_equal & labels_equal

		return mask




	def _get_anchor_negative_triplet_mask(self, labels):
		"""Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
		Args:
			labels: torch tensor with shape [batch_size]
		Returns:
			mask: torch tensor with shape [batch_size, batch_size]
		"""
		# Check if labels[i] != labels[k]
		# Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
		labels_equal = labels.unsqueeze(0).eq(labels.unsqueeze(1))

		mask = ~(labels_equal)

		return mask




	def _batch_all_triplet_loss(self, labels, embeddings, margin, squared=False):
		"""Build the triplet loss over a batch of embeddings.
		We generate all the valid triplets and average the loss over the positive ones.
		Args:
			labels: labels of the batch, of size (batch_size,)
			embeddings: tensor of shape (batch_size, embed_dim)
			margin: margin for triplet loss
			squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
					 If false, output is the pairwise euclidean distance matrix.
		Returns:
			triplet_loss: scalar tensor containing the triplet loss
		"""
		# Get the pairwise distance matrix
		pairwise_dist = self._pairwise_distances(embeddings, squared=squared)

		# shape (batch_size, batch_size, 1)
		anchor_positive_dist = pairwise_dist.unsqueeze(2)
		assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
		# shape (batch_size, 1, batch_size)
		anchor_negative_dist = pairwise_dist.unsqueeze(1)
		assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

		# Compute a 3D tensor of size (batch_size, batch_size, batch_size)
		# triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
		# Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
		# and the 2nd (batch_size, 1, batch_size)
		triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

		# Put to zero the invalid triplets
		# (where label(a) != label(p) or label(n) == label(a) or a == p)
		mask = self._get_triplet_mask(labels)
		mask = mask.float()

		triplet_loss = mask * triplet_loss

		# Remove negative losses (i.e. the easy triplets)
		triplet_loss = torch.clamp(triplet_loss, min=0.0)

		# Count number of positive triplets (where triplet_loss > 0)
		valid_triplets = triplet_loss.gt(1e-16).float()
		num_positive_triplets = valid_triplets.sum()
		num_valid_triplets = mask.sum()
		fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

		# Get final mean triplet loss over the positive valid triplets
		triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

		return triplet_loss, fraction_positive_triplets, num_positive_triplets




	def _batch_hard_triplet_loss(self, labels, embeddings, margin, squared=False):
		"""Build the triplet loss over a batch of embeddings.
		For each anchor, we get the hardest positive and hardest negative to form a triplet.
		Args:
			labels: labels of the batch, of size (batch_size,)
			embeddings: torch tensor of shape (batch_size, embed_dim)
			margin: margin for triplet loss
			squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
					 If false, output is the pairwise euclidean distance matrix.
		Returns:
			triplet_loss: scalar tensor containing the triplet loss
		"""
		# Get the pairwise distance matrix
		pairwise_dist = self._pairwise_distances(embeddings, squared=squared)

		# For each anchor, get the hardest positive
		# First, we need to get a mask for every valid positive (they should have same label)
		mask_anchor_positive = self._get_anchor_positive_triplet_mask(labels)
		mask_anchor_positive = mask_anchor_positive.float()

		# We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
		anchor_positive_dist = mask_anchor_positive * pairwise_dist

		# shape (batch_size, 1)
		hardest_positive_dist = torch.max(anchor_positive_dist, dim=1, keepdim=True)[0]

		# For each anchor, get the hardest negative
		# First, we need to get a mask for every valid negative (they should have different labels)
		mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels)
		mask_anchor_negative = mask_anchor_negative.float()

		# We add the maximum value in each row to the invalid negatives (label(a) == label(n))
		max_anchor_negative_dist = torch.max(pairwise_dist, dim=1, keepdim=True)[0]
		anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

		# shape (batch_size,)
		hardest_negative_dist = torch.min(anchor_negative_dist, dim=1, keepdim=True)[0]

		# Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
		triplet_loss = torch.max(hardest_positive_dist - hardest_negative_dist + margin, torch.zeros_like(hardest_positive_dist))
		# Get final mean triplet loss
		triplet_count = triplet_loss.shape[0]
		triplet_loss = torch.mean(triplet_loss)

		return triplet_loss, triplet_count




	def train(self, batch_hard=False):
		# Initializing the required parameters for keeping track of training
		losses = util.AverageMeter()
		total_triplets = util.AverageMeter()

		# switch to training mode
		self.model.train()

		# training
		for batch_index, images in tqdm(enumerate(self.train_loader)):
			batch_images, labels = self._get_batch_images(images, self.positive_image_dict, transforms=config.data_transforms["train"])
			embeddings = self.model(batch_images)
			if not batch_hard:
				loss, _, triplet_count = self._batch_all_triplet_loss(labels, embeddings, config.TRIPLET_MARGIN)
			else:
				loss, triplet_count = self._batch_hard_triplet_loss(labels, embeddings, config.TRIPLET_MARGIN)

			# measure loss
			losses.update(loss.item(), triplet_count)
			total_triplets.update(triplet_count)

			# compute gradient and do optimizer step
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			if batch_index % 10 == 0:
				print('Train Progress:\t Train Epoch: {} [{}/{}]\t'
					'Loss: {:.4f} ({:.4f}) \t'
					'Total Triplets: {}\t'.format(
					self.epoch, batch_index * config.PARAMS["batch_size"], len(self.train_loader.dataset),
					losses.val, losses.avg, total_triplets.sum))


	def validate(self, batch_hard=False):
		# Initializing the required parameters for keeping track of training
		losses = util.AverageMeter()
		total_triplets = util.AverageMeter()

		# switch to evaluation mode
		self.model.eval()

		with torch.no_grad():
			for batch_index, images in tqdm(enumerate(self.val_loader)):
				batch_images, labels = self._get_batch_images(images, self.positive_image_dict, transforms=config.data_transforms["val"])
				embeddings = self.model(batch_images)
				if not batch_hard:
					loss, _, triplet_count = self._batch_all_triplet_loss(labels, embeddings, config.TRIPLET_MARGIN)
				else:
					loss, triplet_count = self._batch_hard_triplet_loss(labels, embeddings, config.TRIPLET_MARGIN)

				# measure loss
				losses.update(loss.item(), triplet_count)
				total_triplets.update(triplet_count)

				if batch_index % 10 == 0:
					print('Validation Progress:\t Average loss: {:.4f}\t'
						'Total Triplets: {}\t'.format(losses.avg, total_triplets.sum))
		return losses.avg