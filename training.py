# Standard library imports
import os
import json
import math
import random

# Third party imports
import glob
import json
import pickle
from PIL import Image, ImageDraw
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from torchsummary import summary

# Local imports
import config
import inference
from modules import model
from modules import train
from modules import util

DEVICE = config.DEVICE

CURRENT_FREEZE_EPOCH = 0
CURRENT_UNFREEZE_EPOCH = 0
BEST_LOSS = 4													


def training(args):
	# declaring global variables
	global BEST_LOSS
	global CURRENT_FREEZE_EPOCH
	global CURRENT_UNFREEZE_EPOCH

	# steps for preparing and splitting the data for training
	with open(config.ARGS.positive_image_dict, "rb") as f:
		positive_image_dict = pickle.load(f)
	IMAGES = list(positive_image_dict.keys())

	# splitting the images to train and validation set
	random.shuffle(IMAGES)
	train_images = IMAGES[:math.floor(len(IMAGES) * 0.8)]
	val_images = IMAGES[math.ceil(len(IMAGES) * 0.8):]


	# loading the pretrained model and changing the dense layer. Initially the convolution layers will be freezed
	base_model = models.resnet50(pretrained=True).to(DEVICE)
	for param in base_model.parameters():
		param.requires_grad = False
	num_ftrs = base_model.fc.in_features
	base_model.fc = nn.Sequential(nn.Linear(num_ftrs, 1024), nn.Linear(1024, 512), nn.Linear(512, 256))
	base_model = base_model.to(DEVICE)
	tnet = model.Tripletnet(base_model).to(DEVICE)

	if(config.ARGS.resume):
		try:
			CURRENT_FREEZE_EPOCH, CURRENT_UNFREEZE_EPOCH, BEST_LOSS, tnet = util.load_checkpoint(config.ARGS.checkpoint_name, tnet)
		except:
			print("not able to load checkpoint because of non-availability")

	# Initializing the loss function and optimizer
	criterion = torch.nn.MarginRankingLoss(margin=config.TRIPLET_MARGIN)
	optimizer = optim.SGD(tnet.parameters(), lr=config.LR, momentum=config.MOMENTUM)

	# # Decay LR by a factor of 0.1 every 7 epochs
	# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	# Printing total number of parameters
	n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
	print('  + Number of params: {}'.format(n_parameters))

	# training for the first few iteration of freeze layers where apart from the dense layers all the other layres are frozen
	if(CURRENT_FREEZE_EPOCH < config.FREEZE_EPOCHS):
		for epoch in range(CURRENT_FREEZE_EPOCH + 1, config.FREEZE_EPOCHS + 1):
			train_class = train.Train(train_images, val_images, positive_image_dict, base_model, criterion, optimizer, epoch)

			# alternatively training batch hard 
			if(epoch % 2 == 0):
				train_class.train(batch_hard=False)
				loss = train_class.validate(batch_hard=False)
			else:
				train_class.train(batch_hard=True)
				loss = train_class.validate(batch_hard=True)

			# remember best loss and save checkpoint
			is_best = loss < BEST_LOSS
			BEST_LOSS = min(loss, BEST_LOSS)
			util.save_checkpoint({
				'current_freeze_epoch': epoch,
				'current_unfreeze_epoch': 0,
				'state_dict': tnet.state_dict(),
				'best_loss': BEST_LOSS,
			}, is_best)
			CURRENT_FREEZE_EPOCH = epoch

			# visualizing the similar image outputs
			inference.inference()

	# Unfreezing the last few convolution layers
	for param in base_model.parameters():
		param.requires_grad = True
	ct = 0
	for name, child in base_model.named_children():
		ct += 1
		if ct < 7:
			for name2, parameters in child.named_parameters():
				parameters.requires_grad = False

	# training the remaining iterations with the last few layers unfrozen
	if(CURRENT_UNFREEZE_EPOCH < config.UNFREEZE_EPOCHS):
		for epoch in range(CURRENT_UNFREEZE_EPOCH + 1, config.UNFREEZE_EPOCHS + 1):
			train_class = train.Train(train_images, val_images, positive_image_dict, base_model, criterion, optimizer, epoch)

			# alternatively training batch hard
			if(epoch % 2 == 0):
				train_class.train(batch_hard=False)
				loss = train_class.validate(batch_hard=False)
			else:
				train_class.train(batch_hard=True)
				loss = train_class.validate(batch_hard=True)

			# remember best loss and save checkpoint
			is_best = loss < BEST_LOSS
			BEST_ACC = min(loss, BEST_LOSS)
			util.save_checkpoint({
				'current_freeze_epoch': CURRENT_FREEZE_EPOCH,
				'current_unfreeze_epoch': epoch,
				'state_dict': tnet.state_dict(),
				'best_loss': BEST_LOSS,
			}, is_best)
			CURRENT_UNFREEZE_EPOCH = epoch

			# visualizing the similar image outputs
			inference.inference(epoch)

