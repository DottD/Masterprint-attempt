import tensorflow as tf
import numpy
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.regularizers import l2
from keras.optimizers import Adam
import argparse
import os
import math
import time
from PIL import Image
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from keras.applications.resnet50 import ResNet50
# Files
import nistdata
from dmp_classifier import classifier


if __name__ == '__main__':
	# Parse command line arguments
	parser = argparse.ArgumentParser(description="Train a classifier for fingerprints")
	parser.add_argument("in", help="Path to the input database directory")
	parser.add_argument("--load", help="Path to the saved model")
	parser.add_argument("-N", "--num-classes", default=5400, type=int, help="Number of classes")
	parser.add_argument("--model-complexity", default=4, type=int, help="Complexity level of the simpler classifier")
	parser.add_argument("--resnet50", action="store_true", default=False, help="Whether to use the ResNet50 classifier or a simpler one")
	args = vars(parser.parse_args())
	image_path = args["in"]
	model_path = args["load"]
	num_classes = args["num_classes"]
	img_shape = (128, 128, 1)
	use_resnet = args["resnet50"]
	net_width_level = args["model_complexity"]
	# Create the model
	if use_resnet:
		model = ResNet50(weights=None, classes=num_classes)
	else:
		model = classifier(input_shape=img_shape, num_classes=num_classes, net_width_level=net_width_level)
	# Load the model 
	model.load_weights(model_path)
	# Load the input image
	image = Image.open(image_path)
	image = image.convert('L')
	image = img_to_array(image)
	image = numpy.expand_dims(image, axis=0)
	# Crop image to model input shape
	x, y = numpy.random.randint(0, img_shape[0], size=2)
	image = image[:, x:x+img_shape[0], y:y+img_shape[1], :]
	# Eventually adapt the image for the resnet
	if use_resnet:
		image = zoom(image, zoom=(1, 224./img_shape[0], 224./img_shape[1], 3), order=0, prefilter=False)
	# Predict over the input
	prediction = model.predict_on_batch(image)
	# Show prediction
	plt.subplot(2, 1, 1)
	plt.imshow(numpy.squeeze(image))
	plt.subplot(2, 1, 2)
	plt.stem(numpy.squeeze(prediction))
	plt.show()