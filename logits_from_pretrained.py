import os
import argparse
import numpy as np
from keras.applications import ResNet50
from my_keras_preproc_image import ImageDataGenerator
import h5py


if __name__ == '__main__':
	checkImgSize = lambda strin: tuple(int(val) for val in strin.split('x'))
	parser = argparse.ArgumentParser(description="Train a classifier for fingerprints")
	parser.add_argument("in", help="Full path to the input database directory")
	parser.add_argument("out", help="Name of the output folder, created in the current directory")
	parser.add_argument("--img-size", default=(128, 128, 1), type=checkImgSize, help="Expected image size in the form 'WxHxD', W=width, H=height, D=depth")
	parser.add_argument("-B", "--batch-size", default=256, type=int, help="Batch size")
	parser.add_argument("-N", "--num-partials", default=100, type=int, help="Number of partial images per finger")
	parser.add_argument("-V", "--num-val-partials", default=10, type=int, help="Number of partial images per finger (validation)")
	args = vars(parser.parse_args())
	# Read input
	db_path = os.path.abspath(os.path.normpath(args["in"]))
	out_dir = os.path.abspath(os.path.normpath(args["out"]))
	img_shape = args["img_size"]
	batch_size = args["batch_size"]
	N = args["num_partials"]
	V = args["num_val_partials"]
	if not os.path.exists(out_dir): os.makedirs(out_dir)
	# Read and augment data
	def preproc_fn(x):
		x -= np.mean(x, keepdims=True)
		x /= (np.std(x, keepdims=True) + 1E-7)
		x = .5 * (np.tanh(.5 * x) + 1) # this is a sigmoid
		return x
	datagen = ImageDataGenerator(
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
		'target_size': (200,200),
		'color_mode':'rgb',
		'crop_shape': img_shape[:-1],
		'class_mode': None,
		'batch_size': batch_size,
		'shuffle': False # otherwise class order would change
	}
	data_provider = datagen.flow_from_directory(db_path, **idg_args)
	num_classes = data_provider.num_classes
	# Build network
	CNN = ResNet50(include_top=False,
		weights='imagenet',
		input_shape=(200,200,3),
		pooling='avg')
	n_logits = CNN.output_shape[1]
	print("Network created -", n_logits, "logits")
	# Start writing the databases
	filename = os.path.join(out_dir, 'bottleneck_features.hdf5')
	with h5py.File(filename, "w") as f:
		print("Output file created as", filename)
		def db_creation(name, rep):
			db = f.create_dataset(name=name, 
				shape=(rep*num_classes, n_logits), dtype='f',
				maxshape=(None, n_logits),
				compression="gzip",
				compression_opts=9)
			db.attrs['num_classes'] = num_classes
			db.attrs['repetitions'] = rep
			print("Created dataset", name, "("+str(rep)+" repetitions)")
			for k in range(rep):
				begin = k * num_classes # inclusive
				end = (k+1) * num_classes # exclusive
				db[begin:end, :] = CNN.predict_generator(data_provider, verbose=1)
				print("Step", k+1, "/", rep, "- slice", str(begin)+':'+str(end), "DONE")
		db_creation('training', N)
		db_creation('validation', V)
