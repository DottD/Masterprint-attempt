import os
import argparse
import progressbar
from datetime import datetime
import numpy
import tensorflow as tf
from keras.layers import Dense
from keras.losses import binary_crossentropy
from keras.metrics import top_k_categorical_accuracy
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ProgbarLogger, TerminateOnNaN, ModelCheckpoint, LearningRateScheduler, LambdaCallback, ReduceLROnPlateau
from keras_contrib.applications.resnet import ResNet
from nist_data_provider import NistDataProvider
from tensorboard_logging import Logger

	
def binary_sparse_softmax_cross_entropy(target, output, from_logits=False):
	"""
	Expects the output of a sigmoid layer, but computes the
	sparse softmax cross entropy.
	"""
	# TF expects logits, Keras expects probabilities.
	if not from_logits:
		# transform from sigmoid back to logits
		_epsilon = tf.convert_to_tensor(1E-7, output.dtype.base_dtype)
		output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
		output = tf.log(output / (1 - output))

	output_shape = output.get_shape()
	targets = tf.cast(tf.reshape(target, [-1]), 'int64')
	logits = tf.reshape(output, [-1, int(output_shape[-1])])
	res = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=targets,
		logits=logits)
	if len(output_shape) >= 3:
		# if our output includes timestep dimension
		# or spatial dimensions we need to reshape
		return tf.reshape(res, tf.shape(output)[:-1])
	else:
		return res

if __name__ == '__main__':
	# Parse command line arguments
	parser = argparse.ArgumentParser(description="Train a classifier for fingerprints")
	parser.add_argument("in", help="Full path to the input database directory")
	parser.add_argument("out", help="Name of the output folder, created in the current directory")
	parser.add_argument("--load", default=None, help="Name of the folder containing the pre-trained model")
	parser.add_argument("-E", "--epochs", default=100, type=int, help="Number of training steps")
	parser.add_argument("--batch-size", default=128, type=int, help="Number of images to feed per iteration")
	parser.add_argument("--img-size", default=(128, 128, 1), 
		type=lambda strin: tuple(int(val) for val in strin.split('x')),
		help="Expected image size in the form 'WxHxD', W=width, H=height, D=depth; H is not used so far")
	parser.add_argument("--save-epochs", default=1, type=int, help="Save checkpoint every this many epochs")
	parser.add_argument("--learning-rate", default=5E-5, type=float, help="Learning rate for Adam optimizer")
	parser.add_argument("--decay-rate", default=0.01, type=float, help="Decay rate for Adam optimizer")
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
	decay_rate = args["decay_rate"]

	# I/O Folders
	db_path = os.path.abspath(os.path.normpath(args["in"]))
	summary_folder = str(datetime.now().isoformat(sep='_', timespec='seconds')).replace(':', '_').replace('-', '_')
	log_dir = os.path.join(os.path.abspath(args["out"]), summary_folder)
	load_path = args["load"]
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	print('Logs will be summarized in ' + log_dir)
	
	# Load data
	train_provider = NistDataProvider(path=db_path, batch_size=batch_size, validation=None)
	valid_provider = NistDataProvider(path=db_path, batch_size=batch_size, validation=None)
	num_classes = train_provider.num_classes
			
	# Eventually load pre-trained weights
	if load_path:
		CNN = load_model(load_path)
		initial_epoch = int(load_path.split('_')[-1].split('.')[0])
	else:
		CNN = ResNet(input_shape=img_shape, classes=num_classes,
					block='basic', residual_unit='v1', repetitions=[2, 2, 2, 2],
		           	initial_filters=4, # This determines the number of parameters
					activation=None, # final activation manually added
					include_top=False, # last dense layer manually added
					input_tensor=None, dropout=0, transition_dilation_rate=(1, 1), initial_strides=(2, 2), initial_kernel_size=(7, 7),
		           	initial_pooling='max', # for imagenet this is correct
					final_pooling='avg', # final average pooling -> output will be (batch_size, ***)
					top='classification') # no effect with include_top set to false
		CNN = Model(inputs=CNN.inputs[0], outputs=Dense(num_classes, activation='sigmoid', kernel_initializer="he_normal")(CNN.outputs[0]))
		CNN.compile(optimizer=Adam(lr=learning_rate, amsgrad=True), 
				loss=binary_sparse_softmax_cross_entropy, # mutually exclusive classes, independent per-class distributions
				metrics=["sparse_categorical_accuracy"])
		initial_epoch = 0
		
	# Initialize a Summary writer
	logger = Logger(log_dir)
	
	# Define learning rate updater
	compute_lr = lambda e: learning_rate * 1./(1. + decay_rate * e)
	
	# Define the end_of_epoch summary operation
	def summary_op(e, logs):
		# Write summary to file
		logger.log_scalar("Validation/sparse_categorical_accuracy_%", logs['val_sparse_categorical_accuracy']*100, e)
		logger.log_scalar("Validation/loss", logs['val_loss'], e)
		logger.log_scalar("Training/sparse_categorical_accuracy_%", logs['sparse_categorical_accuracy']*100, e)
		logger.log_scalar("Training/loss", logs['loss'], e)
		logger.log_scalar("Model/learning_rate", compute_lr(e), e)
		weights = [y for layer in CNN.layers for x in layer.get_weights() for y in x.flatten().tolist()]
		logger.log_histogram("Model/weights", weights, e)
		logger.log_histogram("Model/weights_no_outlier", weights, e, keep=95)
	
	# List of callbacks
	callbacks = [
		TerminateOnNaN(),
		ModelCheckpoint(os.path.join(log_dir, 'save_'+summary_folder+'_{epoch:d}.h5'),
			monitor='val_loss', 
			verbose=0,
			save_best_only=True,
			save_weights_only=False,
			mode='min',
			period=args["save_epochs"]),
		LearningRateScheduler(compute_lr, verbose=0),
		ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, verbose=1, mode='min', epsilon=0.0001, cooldown=0, min_lr=0),
		LambdaCallback(on_epoch_end=summary_op)
	]
	
	# Create a file with the model description
	with open(os.path.join(log_dir, 'summary_'+summary_folder+'.txt'), mode='w') as F:
		print2F = lambda s: F.write(s+'\n')
		print2F('------')
		print2F("Parameters:")
		for (key, val) in args.items():
			print2F(str(key)+' = '+str(val))
		print2F('------')
		print2F('Logs will be summarized in ' + log_dir)
		CNN.summary(print_fn=print2F)
	
	# Training
	CNN.fit_generator(train_provider, 
		steps_per_epoch=None, # if unspecified, will use the len(generator) as a number of steps
		epochs=nb_epoch,
		verbose=0,
		callbacks=callbacks,
		validation_data=valid_provider,
		validation_steps=None, # if unspecified, will use the len(validation_data) as a number of steps
		class_weight=None,
		max_queue_size=10,
		workers=2,
		use_multiprocessing=True,
		shuffle=False, # false because the generator provides the batches randomly
		initial_epoch=initial_epoch)