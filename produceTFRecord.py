"""
Convert the input database of images to a TFRecord file with given name.
"""
import argparse
import os
import sys
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def input_pipeline(batch_size=64):
	"""
	Creates an input pipeline that reads images and labels from a TFRecord.
	The total number of images is read from filename.
	"""
	# Define the lambda that will decode a serialized example
	def decode_example(serialized_example):
		features={
			'height': tf.FixedLenFeature([], tf.int64),
			'width': tf.FixedLenFeature([], tf.int64),
			'depth': tf.FixedLenFeature([], tf.int64),
			'image_raw': tf.FixedLenFeature([], tf.string),
			'label': tf.FixedLenFeature([], tf.int64),
		}
		parsed_features = tf.parse_single_example(serialized_example, features=features)
		image = tf.decode_raw(parsed_features["image_raw"], tf.float32)
		image_dim = tf.stack([parsed_features["width"], parsed_features["height"], parsed_features["depth"]], axis=0)
		image = tf.reshape(image, image_dim)
		return image, parsed_features["label"]
	# Create a dataset from the given TFRecord filename
	filename = tf.placeholder(tf.string, shape=(), name="filename")
	dataset = tf.data.TFRecordDataset(filename)
	dataset = dataset.map(decode_example)  # Parse the record into tensors.
	dataset = dataset.shuffle(buffer_size=10000)
	dataset = dataset.batch(batch_size)
	iterator = dataset.make_initializable_iterator()
	# Retrieve total number of elements in dataset from filename
	noext = tf.string_split([filename], '.').values[0]
	tot = tf.string_split([noext], '_').values[-1]
	tot_num = tf.string_to_number(tot, tf.int32)
	
	return iterator, filename, tot_num

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
	
def label_sd09(filename):
	"""
	Create a unique label from a file name,
	according to the NIST SD09 naming scheme
	"""
	basename = os.path.basename(filename)
	code = basename.split('_')[1]
	num = code[:-1]
	character = code[-1]
	label = 2 * int(num) - (character == 'f')
	return label

def convert_to(dataset, name, folder, filename2label_fn=label_sd09):
	"""Converts a dataset to tfrecords."""
	# Output file name
	out_dirname = os.path.normpath(folder)
	outname = os.path.join(out_dirname, name + '_' + str(len(dataset)) + '.tfrecords')
	if not os.path.exists(out_dirname):
			os.makedirs(out_dirname)
	print('Writing to', outname)
	writer = tf.python_io.TFRecordWriter(outname)
	# Loop over the list of filenames
	for filename in dataset:
		# Generate the label
		label = filename2label_fn(filename)
		# Read image
		img = np.array(plt.imread(filename))
		img = np.expand_dims(img, axis=2)
		rows = img.shape[0]
		cols = img.shape[1]
		depth = img.shape[2]
		image_raw = img.tostring()
		example = tf.train.Example(features=tf.train.Features(feature={
				'height': _int64_feature(rows),
				'width': _int64_feature(cols),
				'depth': _int64_feature(depth),
				'label': _int64_feature(int(label)),
				'image_raw': _bytes_feature(image_raw)}))
		writer.write(example.SerializeToString())
		print('Written', filename)
	writer.close()

def scan_dir(path):
	"""Recursive scan of the database folder"""
	file_list = []
	for curr_dir, local_dirs, local_files in os.walk(path):
		# filter local files
		local_files = [os.path.join(path,x) for x in local_files if x.endswith('.png')]
		# append to global list
		file_list += local_files
		# recursively scan each subdirectory
		for inner_dir in local_dirs:
			file_list += scan_dir(inner_dir)
	return file_list


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description="Convert the input database of images to a TFRecord file with given name"
	)
	parser.add_argument(
		'--in',
		type=str,
		required=True,
		help='Directory where the images are'
	)
	parser.add_argument(
		'--out',
		type=str,
		required=True,
		help='Directory where the TFRecord will be saved'
	)
	parser.add_argument(
		'--name',
		type=str,
		required=True,
		help='Name of the output file'
	)
	parser.add_argument(
		'--valid-fraction',
		type=float,
		default=0.1,
		help='Number of images saved as validation dataset, as fraction of the total amount of images'
	)
	args = vars(parser.parse_args())
	files = scan_dir(args["in"])
	shuffle(files)
	train_amount = round(len(files)*(1-args["valid_fraction"]))
	train_files = files[:train_amount]
	valid_files = files[train_amount:]
	convert_to(train_files, "training_" + args["name"], args["out"])
	convert_to(valid_files, "validation_" + args["name"], args["out"])