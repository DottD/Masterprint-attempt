import tensorflow as tf
import numpy
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, Dense
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras.activations import sigmoid
from keras.utils import to_categorical
from scipy.ndimage import zoom
import argparse
import os
import math
import time
import progressbar
# Files
import nistdata
from raghakot_resnet import ResnetBuilder
from tensorboard_logging import Logger


def classifier(input_shape=(128, 128, 1),
				num_classes=1000,
				net_width_level=4):
	# Compute the number of convolutional layers, based on input size
	# The last layer must produce 4x4 matrices
	max_rep = math.floor(math.log2(input_shape[0]))-2
	# Compute the initial number of outputs
	init_out_no = 2**net_width_level
	# Add the convolutional layers to the model
	model = Sequential()
	model.add(Conv2D(init_out_no, kernel_size=3, strides=2, activation="relu",
					  input_shape=input_shape,
					  kernel_regularizer=l2(0.01),
					  bias_regularizer=l2(0.01)))
	for n in range(1, max_rep):
		out_no = init_out_no*(2**n)
		model.add(Conv2D(out_no, kernel_size=3, strides=2, activation="relu", 
				  kernel_regularizer=l2(0.01),
				  bias_regularizer=l2(0.01)))
	# Each 4x4 output is reshaped into 1x16 vectors
	model.add(Flatten())
	# Add the final layer that will produce the logits
	# A sigmoid activation function is used to generate independent predictions
	model.add(Dense(num_classes,
			activation="sigmoid",
			kernel_regularizer=l2(0.01),
			bias_regularizer=l2(0.01)))
	return model
	
def serializeHist(values, bins, tag):
	"""Logs the histogram of a list/vector of values."""
	# Convert to a numpy array
	values = numpy.array(values)
	# Create histogram using numpy        
	counts, bin_edges = numpy.histogram(values, bins=bins, range=(0, bins))
	# Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
	# See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
	# Thus, we drop the start of the first bin
	#bin_edges = bin_edges[1:]
	counts = numpy.insert(counts, 0, 0.0)
	# Fill fields of histogram proto
	hist = tf.HistogramProto()
	hist.min = bin_edges[0]
	hist.max = bin_edges[-1]
	hist.num = values.size
	hist.sum = float(numpy.sum(values))
	hist.sum_squares = float(numpy.sum(values**2))
	# Add bin edges and counts
	for edge in bin_edges:
		hist.bucket_limit.append(edge)
	for c in counts:
		hist.bucket.append(c)
	# Create serialized object
	return tf.Summary.Value(tag=tag, histo=hist)
	
if __name__ == '__main__':
	# Parse command line arguments
	parser = argparse.ArgumentParser(description="Train a classifier for fingerprints")
	parser.add_argument("in", help="Full path to the input database directory")
	parser.add_argument("out", help="Name of the output folder, created in the current directory")
	parser.add_argument("--load", default=None, help="Name of the folder containing the pre-trained model")
	parser.add_argument("-E", "--epochs", default=600, type=int, help="Number of training steps")
	parser.add_argument("--batch-size", default=64, type=int, help="Number of images to feed per iteration")
	parser.add_argument("--img-size", default=(128, 128, 1), 
		type=lambda strin: tuple(int(val) for val in strin.split('x')),
		help="Expected image size in the form 'WxHxD', W=width, H=height, D=depth; H is not used so far")
	parser.add_argument("-S", "--summary-epochs", default=1, type=int, help="Summary every this many epochs")
	parser.add_argument("--save-epochs", default=1, type=int, help="Save checkpoint every this many epochs")
	parser.add_argument("--learning-rate", default=5E-5, type=float, help="Learning rate for Adam optimizer")
	parser.add_argument("--model-complexity", default=4, type=int, help="Complexity level of the simpler classifier")
	parser.add_argument("--resnet", action="store_true", default=False, help="Whether to use the ResNet50 classifier or a simpler one")
	args = vars(parser.parse_args())
	print('------')
	print("Parameters:")
	for (key, val) in args.items():
		print(key, '=', val)
	print('------')
	
	# Set parameters from cmd line arguments
	img_shape = args["img_size"]
	img_size = img_shape[0]
	nb_epoch = args["epochs"]
	batch_size = args["batch_size"]
	learning_rate = args["learning_rate"]
	net_width_level = args["model_complexity"]
	use_resnet = args["resnet"]

	# I/O Folders
	db_path = os.path.abspath(os.path.normpath(args["in"])) # Path to the database folder
	out_folder = args["out"] # Name of the folder with training outputs
	load_path = args["load"]
	log_dir = os.path.abspath(os.path.join(os.path.dirname(db_path), out_folder))
	output_log_file = os.path.join(log_dir, 'output.txt')
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	print('Logs will be summarized in ' + log_dir)
	
	# Load data
	def tanh_transform(x):
		return x*2/255 - 1
	X_train, Y_train = nistdata.load_data(s=256, dirname=db_path)
	nb_img = X_train.shape[0]
	num_classes = Y_train.max()+1
	datagen = nistdata.DataGenerator(crop_size=img_size, preprocessing_function=tanh_transform)
	gen_batch = datagen.flow_random(X=X_train, y=Y_train, batch_size=batch_size)

	# Create and compile models
	if use_resnet:
		# resnet18 with sigmoid final activation layer
		channel_first_shape = (img_shape[2], img_shape[0], img_shape[1])
		CNN = ResnetBuilder.build(channel_first_shape, num_classes, "bottleneck", [2, 2, 2, 2], activation="sigmoid")
	else:
		CNN = classifier(input_shape=img_shape, num_classes=num_classes, net_width_level=net_width_level)
	CNN.compile(optimizer=RMSprop(lr=learning_rate), 
			loss="binary_crossentropy", # not mutually exclusive classes, independent per-class distributions
			metrics=["categorical_accuracy"]) # only after a masterprint multiple classes can be activated
			
	# Eventually load pre-trained weights
	if load_path:
		CNN.load_weights(load_path)
		
	# Initialize a Summary writer
	logger = Logger(os.path.join(log_dir, 'summary'))
	
	# Training
	for e in range(1, nb_epoch+1):
		# Compute the number of batch per epoch
		n_batch_per_epoch = math.ceil(nb_img / batch_size)
		epoch_size = n_batch_per_epoch * batch_size
		# Initialize the progress bar
		pb = progressbar.ProgressBar(widgets=[
				'Epoch '+str(e)+'/'+str(nb_epoch)+' ',
				progressbar.widgets.SimpleProgress(format=u'Batch %(value_s)s/%(max_value_s)s'), ' ',
				progressbar.widgets.Bar(marker=u'\u2588'), ' ',
				progressbar.widgets.Timer(), ' ',
				progressbar.widgets.AdaptiveETA()])
		
		for batch_counter in pb(range(n_batch_per_epoch)):
			# Load the batch of images
			X_batch, Y_batch = gen_batch.next()
			Y_batch = to_categorical(Y_batch, num_classes)
			# Update the CNN
			CNN.train_on_batch(X_batch, Y_batch)

		# Print epoch summary (every *** epochs)
		if(e % args["summary_epochs"] == 0):
			pb = progressbar.ProgressBar(widgets=[
					' -- Evaluation ',
					progressbar.widgets.SimpleProgress(format=u'Batch %(value_s)s/%(max_value_s)s'), ' ',
					progressbar.widgets.Bar(marker=u'\u2588'), ' ',
					progressbar.widgets.Timer(), ' ',
					progressbar.widgets.AdaptiveETA()])
			# Evaluate the model
			accuracy = 0.0
			loss = 0.0
			for batch_counter in pb(range(n_batch_per_epoch)):
				# Load the batch of images
				X_batch, Y_batch = gen_batch.next()
				Y_batch = to_categorical(Y_batch, num_classes)
				# Generate prediction
				loc_loss, loc_accuracy = CNN.test_on_batch(X_batch, Y_batch)
				loss += loc_loss
				accuracy += loc_accuracy
			loss /= n_batch_per_epoch
			accuracy /= n_batch_per_epoch
			# Write summary to file
			logger.log_scalar("Evaluation/accuracy", accuracy*100.0, e)
			logger.log_scalar("Evaluation/loss", loss, e)

		# Save model weights (every *** epochs)
		if(e % args["save_epochs"] == 0):
			CNN.save_weights(os.path.join(log_dir, 'CNN_save.h5'), overwrite=True)
