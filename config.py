# Third party imports
import torch
from torchvision import transforms

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data preprocessing details
data_transforms = {
	'train': transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'val': transforms.Compose([
		transforms.Resize(224),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
}

# train related values
LR = 0.001													
MOMENTUM = 0.2

FREEZE_EPOCHS = 1
UNFREEZE_EPOCHS = 200

TRIPLET_MARGIN = 0.3								# the margin to be used for triplet loss
PARAMS = {'batch_size': 8,
			'shuffle': True,
			'num_workers': 16}

EMBEDDING_SIZE = 256

# command line arguments
ARGS = {}