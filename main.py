# Standard library imports
import os
import glob
import argparse

# Third party library imports
import torch
import torch.nn as nn
from torchvision import models

# Local imports
import config
import inference
import training

def main(args):
	# updating all the global variables based on the input arguments
	if(args.freeze_epochs):
		config.FREEZE_EPOCHS = args.freeze_epochs
	if(args.unfreeze_epochs):
		config.UNFREEZE_EPOCHS = args.unfreeze_epochs

	# updating batch size
	if(args.batch_size):
		config.PARAMS["batch_size"] = args.batch_size

	# updating command line arguments to the ARGS variable
	config.ARGS = args

	# calling required functions based on the input arguments
	if args.mode == "inference":
		inference.inference()
	else:
		training.training(args)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument(
		"--mode",
		type=str,
		default="inference",
		help="inference mode or training mode")

	# arguments for training
	parser.add_argument(
		"--batch_size",
		type=int,
		default=128,
		help="the batch_size for training as well as for inference")
	parser.add_argument(
		"--freeze_epochs",
		type=int,
		default=1,
		help="the total number of epochs for which the initial few layers will be frozen")
	parser.add_argument(
		"--unfreeze_epochs",
		type=int,
		default=200,
		help="the total number of epochs for which the full network will be unfrozen")
	parser.add_argument(
		"--resume",
		type=bool,
		default=True,
		help="Flag to resume the training from where it was stopped")
	parser.add_argument(
		"--checkpoint_name",
		type=str,
		default="checkpoint.pth",
		help="the name of the checkpoint file where the weights will be saved")
	parser.add_argument(
		"--positive_image_dict",
		type=str,
		default="./zalando/cropped_images/top/positive_image_dict.pkl",
		help="the dictionary containing information about positive images in this format: {'anchor_image':['postive_image_1', 'positive_image_2']")
	parser.add_argument(
		"--image_path",
		type=str,
		default="./zalando/cropped_images/top/images/",
		help="the directory which has all the images on which the training will be done")

	# arguments for inference
	parser.add_argument(
		"--proximities_from",
		type=list,
		help="list of images from which the similar images have to be shown")
	parser.add_argument(
		"--proximities_from_path",
		type=str,
		default="./zalando/cropped_images/top/images/",
		help="the directory containing the image collection from which the similar images have to be shown")
	parser.add_argument(
		"--proximities_for",
		type=list,
		help="list of images for which the similar images have to be shown")
	parser.add_argument(
		"--proximities_for_path",
		type=str,
		default="./zalando/cropped_images/top/images/",
		help="the directory of images for which the similar images have to be shown")
	parser.add_argument(
		"--top_count",
		type=int,
		default=50,
		help="the number of similar images to be shown")
	parser.add_argument(
		"--inference_output_path",
		type=str,
		default="./output/results/",
		help="the output directory where the inference output as images will be stored")

	main(parser.parse_args())