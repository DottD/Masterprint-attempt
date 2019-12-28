#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import argparse
from time import time
from datetime import datetime
import numpy as np
try:
	import tensorflow_gpu as tf
except:
	import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, ELU, Dropout, BatchNormalization
from keras.losses import binary_crossentropy
from keras.metrics import top_k_categorical_accuracy
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.utils import HDF5Matrix
from keras.callbacks import Callback, ProgbarLogger, TerminateOnNaN, ModelCheckpoint, LearningRateScheduler, LambdaCallback, ReduceLROnPlateau, EarlyStopping
from keras.applications import ResNet50
from tensorboard_logging import Logger
from thumb_from_sd09 import scan_dir
import h5py
from kerastoolbox.callbacks import TelegramMonitor, PrintMonitor

	
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
	def __init__(self, path, args=None, events_dir=None, max_step=None, save_period=10):
		self.save_period = save_period
		self.path = path
		train_dir = os.path.join(path, 'training')
		if not os.path.exists(train_dir): os.makedirs(train_dir)
		self.train_logger = Logger(train_dir)
		valid_dir = os.path.join(path, 'validation')
		if not os.path.exists(valid_dir): os.makedirs(valid_dir)
		self.valid_logger = Logger(valid_dir)
		if args:
			text = 'Parameters\n---------\n'
			for (key, val) in args.items():
				text += '- '+key+' = '+str(val)+'\n'
			self.train_logger.log_text('Description', text)
			self.valid_logger.log_text('Description', text)
		if events_dir and max_step:
			events_files = [F for F in scan_dir(events_dir, '')[1] if os.path.basename(F).startswith('events')]
			for events_file in events_files:
				parent_dir = os.path.dirname(events_file).split(os.sep)[-1]
				if 'training' == parent_dir:
					train_events_file = events_file
				elif 'validation' == parent_dir:
					valid_events_file = events_file
			self.train_logger.copyFrom(train_events_file, max_step=max_step)
			self.valid_logger.copyFrom(valid_events_file, max_step=max_step)
	def on_epoch_begin(self, epoch, logs={}):
		self.starttime=time()
	def on_epoch_end(self, epoch, logs={}):
		self.train_logger.log_scalar("Speed", time()-self.starttime, epoch)
		self.train_logger.log_scalar("sparse_categorical_accuracy_%", logs['sparse_categorical_accuracy']*100, epoch)
		self.train_logger.log_scalar("loss", logs['loss'], epoch)
		self.valid_logger.log_scalar("Speed", time()-self.starttime, epoch)
		self.valid_logger.log_scalar("sparse_categorical_accuracy_%", logs['val_sparse_categorical_accuracy']*100, epoch)
		self.valid_logger.log_scalar("loss", logs['val_loss'], epoch)
		# Model save
		if ((epoch+1) % self.save_period) == 0:
			self.model.save(os.path.join(self.path, 'save_'+str(epoch)+'.h5'))
			_, oldsaves = scan_dir(self.path, '.h5')
			for save in oldsaves:
				try:
					if int(save.split('.')[-2].split('_')[-1]) < epoch:
						os.remove(save)
				except: continue

if __name__ == '__main__':
	tuplify = lambda fn: lambda str_: [fn(x) for x in str_.split('/')]
	checkArch = lambda strin: tuple(int(val) for val in strin.split('-'))
	checkImgSize = lambda strin: tuple(int(val) for val in strin.split('x'))
	# Parse command line arguments
	parser = argparse.ArgumentParser(description="Train a classifier for fingerprints")
	parser.add_argument("in", help="Full path to the input database file (HDF5)")
	parser.add_argument("out", help="Name of the output folder, created in the current directory")
	parser.add_argument("--load", default=None, help="Name of the folder containing the pre-trained model")
	parser.add_argument("--save-epochs", default=50, type=int, help="Save checkpoint every this many epochs")
	parser.add_argument("-E", "--epochs", default=200, type=int, help="Number of training steps")
	# Batch run support
	# Batch run support
	parser.add_argument("-B", "--batch-size", default=[64], type=tuplify(int), help="Number of images to feed per iteration (support batch run through '/' separator)")
	parser.add_argument("-L", "--learning-rate", default=[1E-6], type=tuplify(float), help="Learning rate for Adam optimizer (support batch run through '/' separator)")
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
	nb_epoch = args["epochs"]
	db_path = os.path.abspath(os.path.normpath(args["in"]))
	load_path = args["load"]
	
	# Load data
	with h5py.File(db_path, "r") as f:
		if 'training' in f.keys() and 'validation' in f.keys():
			train_db = f['training']
			valid_db = f['validation']
			if 'num_classes' in train_db.attrs and 'repetitions' in train_db.attrs:
				train_N, train_num_classes = train_db.attrs['repetitions'], train_db.attrs['num_classes']
			else: raise ValueError("The training dataset lacks 'num_classes' and 'repetitions' attributes")
			if 'num_classes' in valid_db.attrs and 'repetitions' in valid_db.attrs:
				valid_N, valid_num_classes = valid_db.attrs['repetitions'], valid_db.attrs['num_classes']
			else: raise ValueError("The validation dataset lacks 'num_classes' and 'repetitions' attributes")
			if train_num_classes != valid_num_classes:
				raise ValueError("The number of classes in training and validation databases differ")
			num_classes = train_num_classes
		else: raise ValueError("The input database lacks training and validation datasets")
	print("Training and validation data loaded")
	print("Training data:", num_classes, "classes repeated", train_N, "times")
	print("Validation data:", num_classes, "classes repeated", valid_N, "times")
		
	train_data = HDF5Matrix(db_path, 'training')
	valid_data = HDF5Matrix(db_path, 'validation')
	train_labels = np.tile(np.arange(num_classes), (train_N,))
	valid_labels = np.tile(np.arange(num_classes), (valid_N,))
	print(train_data.shape, train_labels.shape)
	print(valid_data.shape, valid_labels.shape)
	
	if train_data.shape[1] != valid_data.shape[1]:
		ValueError("Different model used for training and validation, not allowed")
	logits_length = train_data.shape[1]
	# Get info about loaded data
	additional_info = {
		'Logits length': logits_length,
		'Number of classes': num_classes,
		'DB training repetitions': train_N,
		'Training samples': train_data.shape[0],
		'DB validation repetitions': valid_N,
		'Validation samples': valid_data.shape[0]
	}
	
	# Define the function that will be executed for each parameter
	def runOnce(**roargs):
		batch_size = roargs["batch_size"]
		learning_rate = roargs["learning_rate"]
		decay_rate = roargs["decay_rate"]
		EarlyStopping_patience = roargs["ES_patience"]
		EarlyStopping_mindelta =  roargs["ES_mindelta"]
		ReduceLROnPlateau_patience = roargs["RLROP_patience"]
		ReduceLROnPlateau_factor = roargs["RLROP_factor"]
		roargs.update(roargs["additional_info"])
		roargs.pop("additional_info", None)

		summary_folder = str(datetime.now().isoformat(sep='_', timespec='seconds')).replace(':', '_').replace('-', '_')
		log_dir = os.path.join(os.path.abspath(args["out"]), summary_folder)
		if not os.path.exists(log_dir): os.makedirs(log_dir)
		print('Created log folder:', log_dir)
		if load_path:
			initial_epoch = int(load_path.split('_')[-1].split('.')[0])
			# Model creation
			model = load_model(load_path, custom_objects={'binary_sparse_softmax_cross_entropy': binary_sparse_softmax_cross_entropy})
		else:
			initial_epoch = 0
			# Model creation
			logits = Input(shape=(logits_length,))
			prediction = logits
#			n_filters_0 = 512
#			for n_filters in range(3):
#				prediction = Dense(int(512/4**n_filters), kernel_initializer="he_normal", kernel_regularizer=l2(0.01))(prediction)
#				prediction = ELU(0.3)(prediction)
#				prediction = BatchNormalization()(prediction)
#				prediction = Dropout(0.5)(prediction)
			prediction = Dense(num_classes, activation='sigmoid', kernel_initializer="he_normal")(prediction)
			model = Model(inputs=logits, outputs=prediction)
			model.compile(optimizer=Adam(lr=learning_rate, amsgrad=True), 
				loss=binary_sparse_softmax_cross_entropy, # mutually exclusive classes, independent per-class distributions
				metrics=["sparse_categorical_accuracy"])
		# Model description
		trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
		non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
		roargs['Total parameters'] = trainable_count+non_trainable_count
		roargs['Trainable parameters'] = trainable_count
		roargs['Non-trainable parameters'] = non_trainable_count
		# Create custom callback
		if load_path: tensorboardCallback = TensorboardCallback(log_dir, roargs, os.path.dirname(load_path), initial_epoch, save_period=args["save_epochs"])
		else: tensorboardCallback = TensorboardCallback(log_dir, roargs, save_period=args["save_epochs"])
		# Save other information about the model
		with open(os.path.join(log_dir, 'summary_'+summary_folder+'.txt'), mode='w') as F:
			print2F = lambda s: F.write(s+'\n')
			print2F('------')
			print2F("Parameters:")
			for (key, val) in roargs.items():
				print2F(str(key)+' = '+str(val))
			print2F('------')
			print2F('Logs will be summarized in ' + log_dir)
			model.summary(print_fn=print2F)
			
		# List of callbacks
		callbacks = [TerminateOnNaN(), tensorboardCallback,
			TelegramMonitor(api_token="546794449:AAGzmfH9Oa6277Vsl2T9hRrGnNHHSpEMsd8", chat_id="41795159", plot_history=1)]
		if decay_rate and decay_rate > 0:
			compute_lr = lambda e: learning_rate * 1./(1. + decay_rate * e)
			callbacks.append(LearningRateScheduler(compute_lr, verbose=0))
		if EarlyStopping_mindelta and EarlyStopping_patience:
			callbacks.append(EarlyStopping(monitor='val_loss', min_delta=EarlyStopping_mindelta, patience=EarlyStopping_patience, verbose=1, mode='min'))
		if ReduceLROnPlateau_factor and ReduceLROnPlateau_patience and ReduceLROnPlateau_factor <  1:
			callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=ReduceLROnPlateau_factor, patience=ReduceLROnPlateau_patience, verbose=1, mode='min', epsilon=0.0001, cooldown=0, min_lr=1e-10))
		
		# Training
		model.fit(x = train_data, 
			y = train_labels,
			batch_size = batch_size,
			epochs=nb_epoch,
			verbose=0,
			callbacks=callbacks,
			validation_data=(valid_data, valid_labels),
			shuffle=False,
			initial_epoch=initial_epoch)

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
									RLROP_factor = RLROP_factor,
									additional_info = additional_info)