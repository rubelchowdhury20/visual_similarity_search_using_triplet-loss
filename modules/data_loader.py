from PIL import Image

import torch
from torch.utils import data
from torch.autograd import Variable


# Data generator class for generating images for training
class TrainDataset(data.Dataset):
	def __init__(self, images):
		self.images = images
		
	def __len__(self):
		return len(self.images)
	
	def __getitem__(self, index):
		return self.images[index]


# Data generator class for generating images for inference
class ImageDataset(data.Dataset):
	def __init__(self, images, data_path, transforms=None):
		self.images = images
		self.data_path = data_path
		self.transforms = transforms
		
	def __len__(self):
		return len(self.images)
	
	def __getitem__(self, index):
		image = Image.open(self.data_path + self.images[index])
		if self.transforms is not None:
			try:
				return self.images[index], self.transforms(image)
			except:
				print(self.images[index])
		return self.images[index], image