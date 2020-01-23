import torch
from torch.utils import data
from torchvision import transforms

import os
import random
import pickle
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from annoy import AnnoyIndex
from skimage.color import deltaE_ciede2000

import config
from modules import data_loader


class Proximities():
	def __init__(self, model, proximities_for, proximities_from, proximities_for_path, proximities_from_path, epoch=0):
		self.model = model
		self.proximities_for = proximities_for
		self.proximities_from = proximities_from
		self.proximities_for_path = proximities_for_path
		self.proximities_from_path = proximities_from_path
		self.epoch = epoch

		# initializing the dataloader for generating images
		self.proximities_for_set = data_loader.ImageDataset(self.proximities_for, self.proximities_for_path, config.data_transforms["val"])
		self.proximities_for_loader = data.DataLoader(self.proximities_for_set, **config.PARAMS)

		self.proximities_from_set = data_loader.ImageDataset(self.proximities_from, self.proximities_from_path, config.data_transforms["val"])
		self.proximities_from_loader = data.DataLoader(self.proximities_from_set, **config.PARAMS)

	def _get_embeddings(self, emb_dataloader):
		embeddings = []				# list to store the embeddings in dict format as name, embedding
		self.model.eval()
		with torch.no_grad():				# no update of parameters
			for image_names, images in tqdm(emb_dataloader):
				images = images.to(config.DEVICE)
				image_embeddings = self.model(images)
				embeddings.extend([{"image": image_names[index], "embedding": embedding} for index, embedding in enumerate(image_embeddings.cpu().data)])
		return embeddings

	def _create_annoy_index(self, image_embeddings):
		img_to_index = {}
		index_to_img = {}
		embedding_size = config.EMBEDDING_SIZE
		annoy_index = AnnoyIndex(embedding_size, metric="euclidean")
		for index, embedding in tqdm(enumerate(image_embeddings)):
			img_to_index[embedding["image"]] = index
			index_to_img[index] = embedding["image"]
			annoy_index.add_item(index, embedding["embedding"])
		annoy_index.build(100)
		return annoy_index, img_to_index, index_to_img

	def _visualize_similar_images(self, anchor_image, similar_images, similar_image_distances, anchor_image_path, similar_images_path):
		rows = int(np.ceil(len(similar_images)/10))
		stamp_w, stamp_h = 400, 400
		sheet = Image.new('RGB', (stamp_w*10, stamp_h*rows), 'white')
		for row in range(rows):
			for col in range(10):
				if row + col == 0:
					stamp = Image.new('RGB', (stamp_w, stamp_h))
					image = Image.open(anchor_image_path + anchor_image)
					image = image.resize((stamp_w, stamp_h), Image.ANTIALIAS)
					d = ImageDraw.Draw(image)
					stamp.paste(image, (0, 0))
					d.rectangle([0, 0, stamp_w+1, stamp_h+1], outline='red')
					sheet.paste(stamp, (0, 0))
				else:
					cur = row*10+col-1
					if cur < len(similar_images):
						stamp = Image.new('RGB', (stamp_w, stamp_h))
						image = Image.open(similar_images_path + similar_images[cur])
						image = image.resize((stamp_w, stamp_h), Image.ANTIALIAS)
						d = ImageDraw.Draw(stamp)
						d.text((0, stamp_w), str(similar_image_distances[cur]), fill='white')
						stamp.paste(image, (0,0))
						sheet.paste(stamp, (col * stamp_w, row*stamp_h))
		sheet.save(os.path.join(config.ARGS.inference_output_path, "image_sheet-epoch{}-anchor-{}".format(str(self.epoch).zfill(3), anchor_image)))

	def generate_proximities(self, save_similar_images=False):
		# generate embeddings for the query images for which the similar images will be calculated
		query_image_embeddings = self._get_embeddings(self.proximities_for_loader)
		# generate embeddings for the pool of images from which the similar images will be retrieved
		pool_image_embeddings = self._get_embeddings(self.proximities_from_loader)

		annoy_index, img_to_index, index_to_img = self._create_annoy_index(pool_image_embeddings)

		similar_images_list = []		# empty list to save similar image names along with anchor image names

		# iterating through query images and getting similar image results for them
		for query in tqdm(query_image_embeddings):
			similar_images = annoy_index.get_nns_by_vector(query["embedding"], config.ARGS.top_count, include_distances=True)
			similar_image_names = [index_to_img[i] for i in similar_images[0]]
			similar_image_distances = similar_images[1]
			self._visualize_similar_images(query["image"], similar_image_names, similar_image_distances, self.proximities_for_path, self.proximities_from_path)
			similar_images_list.append({"img_name": query["image"], "similar_images": similar_image_names[:config.ARGS.top_count]})

		if(save_similar_images):
			with open("similar_images_list.pkl", "wb") as f:
				pickle.dump(similar_images_list, f)

		similar_images_list = []





	