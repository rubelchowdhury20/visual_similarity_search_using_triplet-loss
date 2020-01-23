# Standard imports
import os
import argparse

# Third party libraries
import glob
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils import data
from torchvision import models

# Local imports
import config
from modules import util
from modules import model
from modules import data_loader

DEVICE = config.DEVICE

def main(args):
	embeddings_for = [os.path.basename(i) for i in glob.glob(args.image_path + "*")]

	# loading the trained model and generating embedding based on that
	base_model = models.resnet50(pretrained=True).to(DEVICE)
	for param in base_model.parameters():
		param.requires_grad = False
	num_ftrs = base_model.fc.in_features
	base_model.fc = nn.Sequential(nn.Linear(num_ftrs, 1024), nn.Linear(1024, 512), nn.Linear(512, 256))
	base_model = base_model.to(DEVICE)
	tnet = model.Tripletnet(base_model).to(DEVICE)

	# loading the trained model with trained weights
	_, _, _, tnet = util.load_checkpoint(args.checkpoint, tnet)

	image_set = data_loader.ImageDataset(embeddings_for, args.image_path, config.data_transforms["val"])
	image_loader = data.DataLoader(image_set, **config.PARAMS)

	embeddings = []				# list to store the embeddings in dict format as name, embedding
	base_model.eval()
	with torch.no_grad():				# no update of parameters
		for image_names, images in tqdm(image_loader):
			images = images.to(config.DEVICE)
			image_embeddings = base_model(images)
			embeddings.extend([{"image": image_names[index], "embedding": embedding} for index, embedding in enumerate(image_embeddings.cpu().data)])


	# saving the embeddings in a pickle file
	with open("./image_embeddings.pkl", "wb") as f:
		pickle.dump(embeddings, f)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument(
		"--checkpoint",
		type=str,
		default="checkpoint.pth",
		help="the name of the checkpoint file where the weights will be saved")

	parser.add_argument(
		"--image_path",
		type=str,
		default="./image_path/",
		help="the directory containing the images for which the embeddings have to be generated.")

	main(parser.parse_args())