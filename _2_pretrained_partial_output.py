#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import time # timing
import argparse
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Dense
from keras.applications import ResNet50
from h5datagen import H5DataGen
import h5py
from shutil import copyfile
from convnetskeras.alexnet import AlexNet 
from kerastoolbox.callbacks import TelegramMonitor, PrintMonitor
from io import StringIO


if __name__ == '__main__':
	checkImgSize = lambda strin: tuple(int(val) for val in strin.split('x'))
	parser = argparse.ArgumentParser(description="Train a classifier for fingerprints")
	parser.add_argument("in", help="Input database file path")
	parser.add_argument("out", default=None, help="Output database file path (append if exists)")
	parser.add_argument("--img-size", default=(128, 128, 1), type=checkImgSize, help="Expected image size in the form 'WxHxD', W=width, H=height, D=depth")
	parser.add_argument("-B", "--batch-size", default=256, type=int, help="Batch size")
	parser.add_argument("-N", "--num-partials", default=100, type=int, help="Number of partial images per finger")
	parser.add_argument("-V", "--num-val-partials", default=10, type=int, help="Number of partial images per finger (validation)")
	parser.add_argument("-A", "--arch", default='alexnet', type=str, help="Network architecture")
	args = vars(parser.parse_args())
	# Set up some monitors
	monitors = [
		TelegramMonitor(api_token="546794449:AAGzmfH9Oa6277Vsl2T9hRrGnNHHSpEMsd8", chat_id="41795159", plot_history=1),
		PrintMonitor()]
	def printmsg(*args):
		output = StringIO()
		print(*args, file=output, end='')
		for monitor in monitors:
			monitor.notify(message=output.getvalue())
		output.close()
	# Read input
	read_mode = "a"
	db_path = os.path.abspath(os.path.normpath(args["in"]))
	filename = os.path.abspath(os.path.normpath(args["out"]))
	out_dir = os.path.dirname(filename)
	if not os.path.exists(out_dir): os.makedirs(out_dir)
	img_shape = args["img_size"]
	batch_size = args["batch_size"]
	N = args["num_partials"]
	V = args["num_val_partials"]
	# Read and augment data
	if args["arch"] == 'resnet':
		target_size = (200,200)
		K.set_image_data_format('channels_last')
	elif args["arch"] == 'alexnet':
		target_size = (227,227)
		K.set_image_data_format('channels_first')
	def preproc_fn(x):
		x -= np.mean(x, keepdims=True)
		x /= (np.std(x, keepdims=True) + 1E-7)
		x = .5 * (np.tanh(.5 * x) + 1) # this is a sigmoid
		return x
	datagen = H5DataGen(
		width_shift_range=0.2,
		height_shift_range=0.2,
		rotation_range = 50, # degrees (int)
		shear_range = 20.*np.pi/180., # radians (float)
		zoom_range = 0.1,
		fill_mode = 'constant',
		cval = 0,
		horizontal_flip = True,
		preprocessing_function = preproc_fn)
	idg_args = {
		'target_size': target_size,
		'color_mode':'rgb',
		'class_mode': None,
		'batch_size': batch_size,
		'shuffle': True
	}
	data_provider = datagen.flow_from_h5file(db_path, **idg_args)
	num_classes = data_provider.num_classes
	printmsg("Found", data_provider.tot_samples, "images belonging to", data_provider.num_classes, "classes.")
	# Build network
	if args["arch"] == 'resnet':
		model = ResNet50(include_top=False,
			weights='imagenet',
			input_shape=(target_size[0],target_size[1],3),
			pooling='avg')
	elif args["arch"] == 'alexnet':
		model = AlexNet(include_top=False,
					weights='imagenet',
					input_shape=(3,target_size[0],target_size[1]),
					trainable=False)
	else:
		raise ValueError("No such architecture implemented")
	model.compile(optimizer='rmsprop', loss='mse')
	n_logits = model.output_shape[1]
	printmsg("Network created -", n_logits, "logits")
	# Start writing the databases
	with h5py.File(filename, read_mode) as f:
		printmsg("Output file", filename, "opened")
		def db_append(name, rep):
			if not name in f.keys():
				printmsg("The "+name+" dataset will be created")
				db = f.create_dataset(name=name, 
					shape=(rep*num_classes, n_logits), dtype='f',
					maxshape=(None, n_logits),
					compression="gzip",
					compression_opts=9)
				db.attrs['num_classes'] = num_classes
				db.attrs['repetitions'] = rep
				prev_rep = 0
				printmsg("Created dataset", name, "("+str(rep)+" repetitions)")
			else:
				db = f[name]
				if (not 'num_classes' in db.attrs) or db.attrs['num_classes'] != num_classes:
					raise ValueError("The dataset lacks 'num_classes' or it differs from input folder traits")
				if (not 'repetitions' in db.attrs):
					raise ValueError("The dataset lacks 'repetitions'")
				prev_rep = db.attrs['repetitions']
				db.attrs['repetitions'] += rep
				db.resize((db.shape[0]+rep*num_classes, db.shape[1]))
				printmsg("Appending logits for", rep, "repetitions of", num_classes, "classes to dataset", name, "(total "+str(db.attrs['repetitions'])+" repetitions)")
			for k in range(prev_rep, db.attrs['repetitions']):
				begin = k * num_classes # inclusive
				end = (k+1) * num_classes # exclusive
				time_start = time.time()
				db[begin:end, :] = model.predict_generator(data_provider, workers=2)
				time_end = time.time()
				printmsg("Step", k+1, "/", db.attrs['repetitions'], "- slice", str(begin)+':'+str(end), "DONE in", time_end-time_start, "s")
		db_append('training', N)
		db_append('validation', V)