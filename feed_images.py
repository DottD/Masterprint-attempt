import os
import psutil
import tensorflow as tf


def feed_images(db_path,
				patterns, 
				ground_truth = False,
				image_size = 128,
				image_depth = 1,
				batch_size = 64,
				num_classes = 5400):
	"""
	Set up an input tf pipeline to load images.
	
	Args:
		db_path: path to the database folder
		patterns: list of file patterns to use
		ground_truth: whether or not returning a ground truth
		image_size: images will be scaled to this size
		image_depth: expected number of channels per image
		batch_size: size of each batch of images
		num_classes: total number of classes (used only if ground_truth is True
	"""
	# Set up the operations to feed the GAN with images
	filename_pattern = os.path.join(db_path, patterns)
	filenames = tf.train.match_filenames_once(filename_pattern)
	filenames = tf.train.string_input_producer(filenames)
	image_reader = tf.WholeFileReader()
	filenames, raw_values = image_reader.read(filenames)
	image = tf.image.decode_png(raw_values)
	image = tf.image.per_image_standardization(image)
	image = tf.image.resize_images(image, [image_size, image_size])
	image.set_shape((image_size, image_size, image_depth))
	min_after_dequeue = 20*batch_size
	num_threads = psutil.cpu_count()
	capacity = min_after_dequeue + num_threads * batch_size
	
	if ground_truth:
		# Process filenames (convention crd_0001f_01_9.png -> 0001f is person's id)
		# 0001 * 2 if 'f', otherwise 0001 * 2 + 1
		basenames =	tf.string_split([filenames], os.sep).values[-1]
		ids = tf.substr(basenames, [4,8], [4,1])
		ids = tf.string_to_number(ids[0], tf.int32) * 2 + tf.cond(tf.equal(ids[1],'s'), 
																lambda: tf.constant(1), 
																lambda: tf.constant(0))
		truth = tf.one_hot(ids, num_classes)
		batch = tf.train.shuffle_batch([image, truth], batch_size, capacity, min_after_dequeue)
	else:
		batch = tf.train.shuffle_batch([image], batch_size, capacity, min_after_dequeue)
																		
	return batch