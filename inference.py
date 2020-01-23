# Standard library imports
import os
import glob
import argparse
import random

# Third party imports
import torch
import torch.nn as nn
from torchvision import models

# Local imports
import config
from modules import util
from modules import model
from modules import proximities

DEVICE = config.DEVICE

def inference(epoch=0):
	# creating the directory to store the inference output results if the directory doesn't exist
	if not os.path.exists(config.ARGS.inference_output_path):
		os.makedirs(config.ARGS.inference_output_path)

	# loading the trained model and generating embedding based on that
	base_model = models.resnet50(pretrained=True).to(DEVICE)
	for param in base_model.parameters():
		param.requires_grad = False
	num_ftrs = base_model.fc.in_features
	base_model.fc = nn.Sequential(nn.Linear(num_ftrs, 1024), nn.Linear(1024, 512), nn.Linear(512, 256))
	base_model = base_model.to(DEVICE)
	tnet = model.Tripletnet(base_model).to(DEVICE)

	# loading the trained model with trained weights
	_, _, _, tnet = util.load_checkpoint(config.ARGS.checkpoint_name, tnet)

	# list of query images for which the similar images will be generated
	if(config.ARGS.proximities_for):
		proximities_for = config.ARGS.proximities_for
	else:
		proximities_for = [os.path.basename(i) for i in glob.glob(config.ARGS.proximities_for_path + "*")]

	if(config.ARGS.proximities_from):
		proximities_from = config.ARGS.proximities_from
	else:
		proximities_from = [os.path.basename(i) for i in glob.glob(config.ARGS.proximities_from_path + "*")]
	
	random.shuffle(proximities_for)
	random.shuffle(proximities_from)
	
	# proximities_for = proximities_for[:10]
	# proximities_from = proximities_from[:10000]

	proximities_class = proximities.Proximities(base_model, proximities_for, proximities_from, config.ARGS.proximities_for_path, config.ARGS.proximities_from_path,epoch)
	proximities_class.generate_proximities(save_similar_images=True)