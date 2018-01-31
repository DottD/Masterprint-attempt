import os
import argparse
import random
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import progressbar
from thumb_from_sd09 import scan_dir

	
class NistDataProvider:
	roi_shape = [128, 128]
	eps = np.finfo(np.float32).eps
	
	# Constructor
	def __init__(self,
				path,
				batch_size=128,
				validation=None,
				ext=".png"):
		self.batch_size = batch_size
		self.path = path
		self.validation = validation
		self.ext = ext
		# Create a list of the images contained in path
		tmp_dirs, self.train_images = scan_dir(self.path, self.ext)
		self.num_classes = len(tmp_dirs)
		# Eventually split the list into train and validation images
		if self.validation:
			count = round(self.validation * len(self.train_images))
			self.validation_images = self.train_images[-count:]
			self.train_images = self.train_images[:-count]
		# Initialize the class on training mode
		self.training_mode = True
		# Compute total available number of batches
		self.length = ceil( len(self.train_images) / self.batch_size )
		
	def change_training_mode(self):
		# Switch from training mode to validation, or the other way round
		if self.validation:
			self.training_mode = not self.training_mode
			if self.training_mode:
				list_of_images = self.train_images
			else:
				list_of_images = self.validation_images
			self.length = ceil( len(list_of_images) / self.batch_size )
		else: 
			raise Exception("Validation should be set to change training mode")
		
	def __iter__(self):
		# Initialize counter
		self.idx = 0
		# Take the proper image path
		if self.training_mode:
			list_of_images = self.train_images
		else:
			list_of_images = self.validation_images
		# Shuffle the list of images
		random.shuffle(list_of_images)
		return self
		
	def __next__(self):
		X = []
		Y = []
		for _ in range(self.batch_size):
			# Take the proper image path
			if self.training_mode:
				list_of_images = self.train_images
			else:
				list_of_images = self.validation_images
			# Check for array's end
			if self.idx >= len(list_of_images):
				# Raise an iteration to stop looping
				raise StopIteration()
			else:
				path = list_of_images[self.idx]
			# Compute label
			label = self.path2label(path)
			# Update counter
			self.idx += 1
			# Load image
			img = np.array(plt.imread(path))
			# Check if the dimension of the image is sufficient
			if img.shape[0] >= self.roi_shape[0] and img.shape[1] >= self.roi_shape[1]:
				# Randomly draw the position of the upper-left corner			
				x, y = [np.random.randint(0, img.shape[i]-self.roi_shape[i]+1) for i in range(2)]
				# Crop the image
				img = img[x:x+self.roi_shape[0], y:y+self.roi_shape[1]]
				# Sample normalization
				img = (img-img.mean())/(img.std()+self.eps)
				# Range fitting
				img = np.tanh(img)
				# Extend to (rows, columns, depth)
				img = np.expand_dims(img, axis=2)
				X.append(img)
				Y.append(label)
		
		return np.array(X), np.array(Y)
		
	def __len__(self):
		return self.length
		
	def path2label(self, path):
		"""
		Create a unique label from a file name,
		according to the NIST SD09 naming scheme
		"""
		path = os.path.basename(path)
		code = path.split('.')[-2].split('_')[1]
		num = code[:-1]
		character = code[-1]
		label = 2 * int(num) - (character == 'f')
		return label-1
		
def to_smooth_categorical(labels, num_classes, range_1=(0.7, 1.2), range_0=(0.0, 0.3)):
	# Get the batch size
	bs = labels.shape[0]
	# Initialize the output matrix to zeros
	Y = np.random.rand(bs, num_classes) * (range_0[1]-range_0[0]) + range_0[0]
	# Generate a random float in the given range for each label
	R = np.random.rand(bs) * (range_1[1]-range_1[0]) + range_1[0]
	# Mark the correct label for each sample
	Y[np.arange(bs), labels] = R
	
	return Y
		
if __name__=='__main__':
	script_descr = "NistDataProvider definition - test"
	print(script_descr)
	parser = argparse.ArgumentParser(description=script_descr)
	parser.add_argument("in", help="Path to the input database file")
	args = vars(parser.parse_args())
	provider = NistDataProvider(args["in"], batch_size=256, validation=0.1)
	# Training
	for X, Y in provider:
		for i in range(X.shape[0]):
			print(X.shape, Y.shape)
			x = X[i, :, :, 0]
			y = Y[i]
			plt.imshow(x)
			plt.title(str(y)+' min='+str(x.min())+' max='+str(x.max()))
			plt.show()