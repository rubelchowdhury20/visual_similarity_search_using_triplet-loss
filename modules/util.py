# Standard library imports
import os

# Third party imports
import shutil
import torch

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


# utility functions for training
def save_checkpoint(state, is_best, filename='checkpoint.pth'):
	"""Saves checkpoint to disk"""
	directory = "./weights/"
	if not os.path.exists(directory):
		os.makedirs(directory)
	filename = directory + filename
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, directory+'model_best.pth')

def load_checkpoint(filename, model):
	"""Loading the weights to the model from the checkpoint	
	Args:
		filename: the checkpoint file name
		model: the model to be loaded with the weights saved as checkpoint
		
	Returns:
		current_freeze_epoch: the number of freeze epochs completed
		current_unfreeze_epoch: the number of unfreeze epochs completed
		best_loss: best loss value till now
		model: the model loaded with weights from the state dict
	"""

	filename = "./weights/" + filename
	if os.path.isfile(filename):
		print("=> loading checkpoint '{}'".format(filename))
		checkpoint = torch.load(filename)

		current_freeze_epoch = checkpoint['current_freeze_epoch']
		current_unfreeze_epoch = checkpoint['current_unfreeze_epoch']

		best_loss = checkpoint['best_loss']

		model.load_state_dict(checkpoint['state_dict'])

		if(current_unfreeze_epoch == 0):
			print("=> loaded checkpoint '{}' (frozen epoch {})".format(filename, current_freeze_epoch))
		else:
			print("=> loaded checkpoint '{}' (unfrozen epoch {})".format(filename, current_unfreeze_epoch))
		return current_freeze_epoch, current_unfreeze_epoch, best_loss, model
	else:
		print("=> no checkpoint found at '{}'".format(filename))

	


