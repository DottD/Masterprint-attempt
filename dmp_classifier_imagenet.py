import os
import argparse
from time import time
from datetime import datetime
from shutil import copyfile
import numpy
try:
	import tensorflow_gpu as tf
except:
	import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from keras.losses import binary_crossentropy
from keras.metrics import top_k_categorical_accuracy
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import Callback, ProgbarLogger, TerminateOnNaN, ModelCheckpoint, LearningRateScheduler, LambdaCallback, ReduceLROnPlateau, EarlyStopping
from keras.applications import ResNet50
from my_keras_preproc_image import ImageDataGenerator
from tensorboard_logging import Logger
from thumb_from_sd09 import scan_dir

	
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

class TensorboardCallback(Callback):
	def __init__(self, logger, args=None):
		self.logger = logger
	def on_epoch_begin(self, epoch, logs={}):
		self.starttime=time()
	def on_epoch_end(self, epoch, logs={}):
		self.logger.log_scalar("Speed", time()-self.starttime, epoch)
		self.logger.log_scalar("Validation/sparse_categorical_accuracy_%", logs['val_sparse_categorical_accuracy']*100, epoch)
		self.logger.log_scalar("Validation/loss", logs['val_loss'], epoch)
		self.logger.log_scalar("Training/sparse_categorical_accuracy_%", logs['sparse_categorical_accuracy']*100, epoch)
		self.logger.log_scalar("Training/loss", logs['loss'], epoch)

if __name__ == '__main__':
	tuplify = lambda fn: lambda str_: [fn(x) for x in str_.split('/')]
	checkArch = lambda strin: tuple(int(val) for val in strin.split('-'))
	checkImgSize = lambda strin: tuple(int(val) for val in strin.split('x'))
	# Parse command line arguments
	parser = argparse.ArgumentParser(description="Train a classifier for fingerprints")
	parser.add_argument("in", help="Full path to the input database directory")
	parser.add_argument("out", help="Name of the output folder, created in the current directory")
	parser.add_argument("--load", default=None, help="Name of the folder containing the pre-trained model")
	parser.add_argument("--img-size", default=(128, 128, 1), type=checkImgSize, help="Expected image size in the form 'WxHxD', W=width, H=height, D=depth")
	parser.add_argument("--save-epochs", default=50, type=int, help="Save checkpoint every this many epochs")
	parser.add_argument("-E", "--epochs", default=200, type=int, help="Number of training steps")
	# Batch run support
	parser.add_argument("-B", "--batch-size", default=[64], type=tuplify(int), help="Number of images to feed per iteration (support batch run through '/' separator)")
	parser.add_argument("-L", "--learning-rate", default=[20E-5], type=tuplify(float), help="Learning rate for Adam optimizer (support batch run through '/' separator)")
	parser.add_argument("-D", "--decay-rate", default=[None], type=tuplify(float), help="Decay rate for Adam optimizer (support batch run through '/' separator)")
	parser.add_argument("--ES-patience", default=[None], type=tuplify(int), help="Early stopping patience (support batch run through '/' separator)")
	parser.add_argument("--ES-mindelta", default=[None], type=tuplify(float), help="Early stopping minimum difference (support batch run through '/' separator)")
	parser.add_argument("--RLROP-patience", default=[None], type=tuplify(int), help="ReduceLROnPlateau patience (support batch run through '/' separator)")
	parser.add_argument("--RLROP-factor", default=[None], type=tuplify(float), help="ReduceLROnPlateau factor (support batch run through '/' separator)")
	print()
	args = vars(parser.parse_args())
	for key, val in args.items():
		print(key, val)
	print()
			
	# Set fixed parameters from cmd line arguments
	img_shape = args["img_size"]
	img_size = img_shape[0]
	nb_epoch = args["epochs"]
	db_path = os.path.abspath(os.path.normpath(args["in"]))
	load_path = args["load"]
	
	# Define the function that will be executed for each parameter
	def runOnce(**roargs):
		batch_size = roargs["batch_size"]
		learning_rate = roargs["learning_rate"]
		decay_rate = roargs["decay_rate"]
		EarlyStopping_patience = roargs["ES_patience"]
		EarlyStopping_mindelta =  roargs["ES_mindelta"]
		ReduceLROnPlateau_patience = roargs["RLROP_patience"]
		ReduceLROnPlateau_factor = roargs["RLROP_factor"]

		# I/O Folders
		summary_folder = str(datetime.now().isoformat(sep='_', timespec='seconds')).replace(':', '_').replace('-', '_')
		log_dir = os.path.join(os.path.abspath(args["out"]), summary_folder)
		if not os.path.exists(log_dir): os.makedirs(log_dir)
		print('Created log folder:', log_dir)
		logger = Logger(log_dir)
		if load_path:
			load_events_file = [F for F in scan_dir(os.path.dirname(load_path), '')[1] if os.path.basename(F).startswith('events')][0]
			initial_epoch = int(load_path.split('_')[-1].split('.')[0])
			logger.copyFrom(load_events_file, max_step=initial_epoch)
		else:
			initial_epoch = 0
		
		# Load data
		# image is augmented, then preproc_fn
		def preproc_fn(x):
			x -= numpy.mean(x, keepdims=True)
			x /= (numpy.std(x, keepdims=True) + 1E-7)
			x = .5 * (numpy.tanh(.5 * x) + 1) # this is a sigmoid
			print("Check", x.min(), x.max())													 ######
			return x
			
		datagen = ImageDataGenerator(
			width_shift_range=0.2,
			height_shift_range=0.2,
			rotation_range = 50, # degrees (int)
			shear_range = 20.*numpy.pi/180., # radians (float)
			zoom_range = 0.1,
			fill_mode = 'constant',
			cval = 0,
			horizontal_flip = True
			preprocessing_function = preproc_fn)
		idg_args = {
			'target_size': (200,200),
			'color_mode':'rgb',
			'crop_shape': img_shape[:-1],
			'class_mode': 'sparse',
			'batch_size': batch_size
		}
		train_provider = datagen.flow_from_directory(db_path, **idg_args)
		valid_provider = datagen.flow_from_directory(db_path, **idg_args)
		
		# Get info about loaded data
		num_classes = train_provider.num_classes
		roargs['Training classes'] = num_classes
		roargs['Training samples'] = train_provider.samples
		roargs['Validation classes'] = valid_provider.num_classes
		roargs['Validation samples'] = valid_provider.samples
				
		# Eventually load pre-trained weights
		if load_path:
			CNN = load_model(load_path, custom_objects={'binary_sparse_softmax_cross_entropy': binary_sparse_softmax_cross_entropy})
		else:
			# Set up the network
			CNN = ResNet50(include_top=False,
				weights='imagenet',
				input_shape=(200,200,3),
				pooling='avg')
			# Freeze pre-trained model
			for layer in CNN.layers: layer.trainable = False
			# Add final fully connected layer
			CNN = Model(inputs=CNN.inputs[0], outputs=Dense(num_classes, activation='sigmoid', kernel_initializer="he_normal")(CNN.outputs[0]))
			# Compilation
			CNN.compile(optimizer=Adam(lr=learning_rate, amsgrad=True), 
					loss=binary_sparse_softmax_cross_entropy, # mutually exclusive classes, independent per-class distributions
					metrics=["sparse_categorical_accuracy"])
				
		# Append the model description to text
		trainable_count = int(
			numpy.sum([K.count_params(p) for p in set(CNN.trainable_weights)]))
		non_trainable_count = int(
			numpy.sum([K.count_params(p) for p in set(CNN.non_trainable_weights)]))
		roargs['Total parameters'] = trainable_count+non_trainable_count
		roargs['Trainable parameters'] = trainable_count
		roargs['Non-trainable parameters'] = non_trainable_count
		# Save other information about the model
		with open(os.path.join(log_dir, 'summary_'+summary_folder+'.txt'), mode='w') as F:
			print2F = lambda s: F.write(s+'\n')
			print2F('------')
			print2F("Parameters:")
			for (key, val) in args.items():
				print2F(str(key)+' = '+str(val))
			print2F('------')
			print2F('Logs will be summarized in ' + log_dir)
			CNN.summary(print_fn=print2F)

		# Log information about the model and the parameters
		text = 'Parameters\n---------\n'
		for (key, val) in args.items():
			text += '- '+key+' = '+str(val)+'\n'
		logger.log_text('Description', text)
			
		# List of callbacks
		callbacks = [TerminateOnNaN(), TensorboardCallback(logger)]
		callbacks.append(ModelCheckpoint(os.path.join(log_dir, 'save_'+summary_folder+'_{epoch:d}.h5'),
			monitor='val_loss', 
			verbose=0,
			save_best_only=False,
			save_weights_only=False,
			mode='min',
			period=args["save_epochs"]))
		if decay_rate and decay_rate > 0:
			compute_lr = lambda e: learning_rate * 1./(1. + decay_rate * e)
			callbacks.append(LearningRateScheduler(compute_lr, verbose=1))
		if EarlyStopping_mindelta and EarlyStopping_patience:
			callbacks.append(EarlyStopping(monitor='val_loss', min_delta=EarlyStopping_mindelta, patience=EarlyStopping_patience, verbose=1, mode='min'))
		if ReduceLROnPlateau_factor and ReduceLROnPlateau_patience and ReduceLROnPlateau_factor <  1:
			callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=ReduceLROnPlateau_factor, patience=ReduceLROnPlateau_patience, verbose=1, mode='min', epsilon=0.0001, cooldown=0, min_lr=0))
		
		# Training
		try:
			CNN.fit_generator(train_provider, 
				steps_per_epoch=None, # if unspecified, will use the len(generator) as a number of steps
				epochs=nb_epoch,
				verbose=2,
				callbacks=callbacks,
				validation_data=valid_provider,
				validation_steps=None, # if unspecified, will use the len(validation_data) as a number of steps
				class_weight=None,
				max_queue_size=10,
				workers=2,
				use_multiprocessing=True,
				shuffle=True, # false because the generator provides the batches randomly
				initial_epoch=initial_epoch)
		except KeyboardInterrupt:
			print("Training interrupted")

	# Set variable parameters from cmd line args
	tests = [len(list_) for list_ in args.values() if isinstance(list_, list)]
	n_tests = 1
	for t in tests: n_tests *= t
	print(n_tests, "total tests will be performed... be patient!")
	for batch_size in args["batch_size"]:
		for learning_rate in args["learning_rate"]:
			for decay_rate in args["decay_rate"]:
				for ES_patience in args["ES_patience"]:
					for ES_mindelta in args["ES_mindelta"]:
						for RLROP_patience in args["RLROP_patience"]:
							for RLROP_factor in args["RLROP_factor"]:
								runOnce(batch_size = batch_size,
									learning_rate = learning_rate,
									decay_rate = decay_rate,
									ES_patience = ES_patience,
									ES_mindelta =  ES_mindelta,
									RLROP_patience = RLROP_patience,
									RLROP_factor = RLROP_factor)