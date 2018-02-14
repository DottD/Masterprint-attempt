import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from dmp_classifier import top_1_categorical_accuracy


def path2label(path):
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

if __name__ == '__main__':
	# Parse command line arguments
	parser = argparse.ArgumentParser(description="Train a classifier for fingerprints")
	parser.add_argument("in", help="Path to the input database directory")
	parser.add_argument("--load", help="Path to the saved model")
	parser.add_argument("-N", "--num-classes", default=5400, type=int, help="Number of classes")
	parser.add_argument("--img-size", default=(128, 128, 1), 
		type=lambda strin: tuple(int(val) for val in strin.split('x')),
		help="Expected image size in the form 'WxHxD', W=width, H=height, D=depth; H is not used so far")
	args = vars(parser.parse_args())
	image_path = args["in"]
	model_path = args["load"]
	num_classes = args["num_classes"]
	img_shape = args["img_size"]
	
	# Load the model 
	model = load_model(model_path, custom_objects={'top_1_categorical_accuracy': top_1_categorical_accuracy})
	
	# Load the input image
	label = path2label(image_path)
	img = np.array(plt.imread(image_path))
	if img.shape[0] >= img_shape[0] and img.shape[1] >= img_shape[1]:
		# Randomly draw the position of the upper-left corner			
		x, y = [np.random.randint(0, img.shape[i]-img_shape[i]+1) for i in range(2)]
		# Crop the image
		img = img[x:x+img_shape[0], y:y+img_shape[1]]
		# Sample normalization
		img = (img-img.mean())/(img.std()+np.finfo(float).eps)
		# Range fitting
		img = np.tanh(img)
		# Extend to (rows, columns, depth)
		img = np.expand_dims(img, axis=2)
	img = np.expand_dims(img, axis=0)

	# Predict over the input
	prediction = model.predict_on_batch(img)
	
	# Show prediction
	plt.subplot(211)
	plt.imshow(np.squeeze(img))
	plt.subplot(212)
	plt.stem(np.squeeze(prediction), marker='.', markersize=0.5, linestyle=' ', color='k')
	plt.stem([label], [prediction.max()], marker='.', markersize=0.5, linestyle=':', color='r')
	plt.title("True label = "+str(label))
	plt.show()