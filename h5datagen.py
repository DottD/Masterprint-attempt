#!/usr/bin/python
# -*- coding: UTF-8 -*-

import random
import numpy as np
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, Iterator, img_to_array
import h5py

try:
	from PIL import Image as pil_image
except ImportError:
	pil_image = None

if pil_image is not None:
	_PIL_INTERPOLATION_METHODS = {
		'nearest': pil_image.NEAREST,
		'bilinear': pil_image.BILINEAR,
		'bicubic': pil_image.BICUBIC,
	}
	# These methods were only introduced in version 3.4.0 (2016).
	if hasattr(pil_image, 'HAMMING'):
		_PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
	if hasattr(pil_image, 'BOX'):
		_PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
	# This method is new in version 1.1.3 (2013).
	if hasattr(pil_image, 'LANCZOS'):
		_PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS
		
def chunks(l, n):
	"""
	Yield successive n-sized chunks from l.
	See https://stackoverflow.com/a/312464/6822728
	"""
	for i in range(0, len(l), n):
		yield l[i:i + n]

class H5DataGen(ImageDataGenerator):
	""" Generate minibatches of image data with real-time data augmentation from an h5 db file. 
	
	# Arguments
			featurewise_center: set input mean to 0 over the dataset.
			samplewise_center: set each sample mean to 0.
			featurewise_std_normalization: divide inputs by std of the dataset.
			samplewise_std_normalization: divide each input by its std.
			zca_whitening: apply ZCA whitening.
			zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
			rotation_range: degrees (0 to 180).
			width_shift_range: fraction of total width, if < 1, or pixels if >= 1.
			height_shift_range: fraction of total height, if < 1, or pixels if >= 1.
			shear_range: shear intensity (shear angle in degrees).
			zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
					in the range [1-z, 1+z]. A sequence of two can be passed instead
					to select this range.
			channel_shift_range: shift range for each channel.
			fill_mode: points outside the boundaries are filled according to the
					given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
					is 'nearest'.
					Points outside the boundaries of the input are filled according to the given mode:
							'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
							'nearest':  aaaaaaaa|abcd|dddddddd
							'reflect':  abcddcba|abcd|dcbaabcd
							'wrap':  abcdabcd|abcd|abcdabcd
			cval: value used for points outside the boundaries when fill_mode is
					'constant'. Default is 0.
			horizontal_flip: whether to randomly flip images horizontally.
			vertical_flip: whether to randomly flip images vertically.
			rescale: rescaling factor. If None or 0, no rescaling is applied,
					otherwise we multiply the data by the value provided. This is
					applied after the `preprocessing_function` (if any provided)
					but before any other transformation.
			preprocessing_function: function that will be implied on each input.
					The function will run before any other modification on it.
					The function should take one argument:
					one image (Numpy tensor with rank 3),
					and should output a Numpy tensor with the same shape.
			data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
					(the depth) is at index 1, in 'channels_last' mode it is at index 3.
					It defaults to the `image_data_format` value found in your
					Keras config file at `~/.keras/keras.json`.
					If you never set it, then it will be "channels_last".
	"""
	def __init__(self, **args):
		""" This init keeps the default parameters of base class
		and delegates it for any initial setup. """
		super().__init__(**args)
		
	def flow_from_h5file(self,h5file, groups=None, 
						 target_size=(256, 256), color_mode='rgb',
						 classes=None, class_mode='categorical',
						 batch_size=32, shuffle=True, seed=None,
						 data_format=None, save_to_dir=None,
						 save_prefix='', save_format='png',
						 interpolation='nearest'):
			return H5FileIterator(
					h5file, self,
					target_size=target_size, color_mode=color_mode,
					classes=classes, class_mode=class_mode,
					batch_size=batch_size, shuffle=shuffle, seed=seed,
					data_format=self.data_format,
					save_to_dir=save_to_dir,
					save_prefix=save_prefix,
					save_format=save_format,
					interpolation=interpolation)
					

class H5FileIterator(Iterator):
	"""Iterator capable of reading data from a h5 file.
	It should contain groups of three datasets, indexed
	the same way. At index k:
		- 'images', has the image of the k-th subject
		- 'classes', has the k-th class name
		- 'topleft', has the array of the possible positions of the ROI's top-left corner
	Each group in this file will be considered to contain suitable data,
	or alternatively you could specify groups via the `groups` argument.
	Each group should have a 'roi_size' attribute, telling the shape of the ROIs
	whose top-left corner can be one of 'topleft' database's elements
	(as a (width, height) tuple).
	Groups are supposed to have different subjects, but there is no check;
	using groups containing the same subject leads to undefined behaviour.
	If you want to specify a subset of classes, you could via the 'classes' argument.
	Images must be saved as 1D float64 array; the actual shape will be recovered
	using the information stored in the Dataset 'rows' at the corresponding index.
	Topleft is a 1D int32 array with the row indices at even positions and
	column indices at odd positions.
	
							
	# Arguments
			h5file: Path to the h5 file to read images from.
			groups: List of groups to use from the h5 file, otherwise
					computed automatically.
			image_data_generator: Instance of `ImageDataGenerator`
					to use for random transformations and normalization.
			target_size: tuple of integers, dimensions to resize input images to.
			color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
			classes: Optional list of strings, names of subdirectories
					containing images from each class (e.g. `["dogs", "cats"]`).
					It will be computed automatically if not set.
			class_mode: Mode for yielding the targets:
					`"binary"`: binary targets (if there are only two classes),
					`"categorical"`: categorical targets,
					`"sparse"`: integer targets,
					`"input"`: targets are images identical to input images (mainly
							used to work with autoencoders),
					`None`: no targets get yielded (only input images are yielded).
			batch_size: Integer, size of a batch.
			shuffle: Boolean, whether to shuffle the data between epochs.
			seed: Random seed for data shuffling.
			data_format: String, one of `channels_first`, `channels_last`.
			save_to_dir: Optional directory where to save the pictures
					being yielded, in a viewable format. This is useful
					for visualizing the random transformations being
					applied, for debugging purposes.
			save_prefix: String prefix to use for saving sample
					images (if `save_to_dir` is set).
			save_format: Format to use for saving sample images
					(if `save_to_dir` is set).
			interpolation: Interpolation method used to resample the image if the
					target size is different from that of the loaded image.
					Supported methods are "nearest", "bilinear", and "bicubic".
					If PIL version 1.1.3 or newer is installed, "lanczos" is also
					supported. If PIL version 3.4.0 or newer is installed, "box" and
					"hamming" are also supported. By default, "nearest" is used.
	"""

	def __init__(self, h5file, image_data_generator, groups=None, 
				 target_size=(256, 256), color_mode='rgb',
				 classes=None, class_mode='categorical',
				 batch_size=32, shuffle=True, seed=None,
				 data_format=None, save_to_dir=None,
				 save_prefix='', save_format='png',
				 interpolation='nearest'):
		# Open h5 file in reading-only mode
		self.h5file = h5py.File(h5file, 'r')
		# Setup the instance's properties according to arguments
		if groups is None:
			groups = list(self.h5file.keys())
		else:
			groups = [grp for grp in groups if (grp in self.h5file.keys() and self.h5file[grp].keys() == {'images', 'classes', 'topleft'})]
		self.sizes = [self.h5file[grp].attrs['roi_size'] for grp in groups if 'roi_size' in self.h5file[grp].attrs]
		self.groups = groups
		if data_format is None:
			data_format = K.image_data_format()
		self.image_data_generator = image_data_generator
		self.target_size = tuple(target_size)
		if color_mode not in {'rgb', 'grayscale'}:
			raise ValueError('Invalid color mode:', color_mode, '; expected "rgb" or "grayscale".')
		self.color_mode = color_mode
		self.data_format = data_format
		if self.color_mode == 'rgb':
			if self.data_format == 'channels_last':
				self.image_shape = self.target_size + (3,)
			else:
				self.image_shape = (3,) + self.target_size
		else:
			if self.data_format == 'channels_last':
				self.image_shape = self.target_size + (1,)
			else:
				self.image_shape = (1,) + self.target_size
		if class_mode not in {'categorical', 'binary', 'sparse', 'input', None}:
				raise ValueError('Invalid class_mode:', class_mode,
								 '; expected one of "categorical", '
								 '"binary", "sparse", "input" or None.')
		self.class_mode = class_mode
		self.save_to_dir = save_to_dir
		self.save_prefix = save_prefix
		self.save_format = save_format
		self.interpolation = interpolation

		# Total number of classes
		self.num_classes = 0
		# Names of those classes
		self.classes = []
		# Total number of available ROIs
		self.tot_samples = 0
		for group_name in groups:
			classes = self.h5file[group_name]['classes']
			self.num_classes += classes.shape[0]
			self.classes.extend(classes)
			self.tot_samples += sum(len(line) for line in self.h5file[group_name]['topleft'])

		super().__init__(self.num_classes, batch_size, shuffle, seed)
	
	def _load_img(self, idx, 
				grayscale=False, 
				target_size=None,
				interpolation='nearest'):
		"""Loads an image into PIL format.
		Get the image from the h5 file, resize it to the given target
		shape using the given interpolation method and return it.

		# Arguments
				class_idx: Index of the image in the h5 file (regardless of group membership)
				grayscale: Boolean, whether to load the image as grayscale.
				target_size: Either `None` (default to original size)
						or tuple of ints `(img_height, img_width)`.
				interpolation: Interpolation method used to resample the image if the
						target size is different from that of the loaded image.
						Supported methods are "nearest", "bilinear", and "bicubic".
						If PIL version 1.1.3 or newer is installed, "lanczos" is also
						supported. If PIL version 3.4.0 or newer is installed, "box" and
						"hamming" are also supported. By default, "nearest" is used.

		# Returns
				A PIL Image instance.

		# Raises
				ImportError: if PIL is not available.
				ValueError: if interpolation method is not supported.
				ValueError: if class_idx is out of range
		"""
		if pil_image is None:
			raise ImportError('Could not import PIL.Image. The use of `array_to_img` requires PIL.')
		if idx >= self.num_classes:
			raise ValueError("Idx out of range")
		# Given the index (spanning among all the groups)
		# get class name, image and list of top-left positions of available ROIs
		k = 0
		for group_idx, group_name in enumerate(self.groups):
			with self.lock:
				group = self.h5file[group_name]
				num_classes = group['classes'].shape[0]
			class_idx = idx-k
			if class_idx < num_classes: # this is the group where it belongs to
				# Load from file
				with self.lock:
					image = group['images'][class_idx]
					topleft = group['topleft'][class_idx]
					rows = group['rows'][class_idx]
				# Get the image in PIL format
				image = np.reshape(image, (rows, -1))
				image = pil_image.fromarray(image)
				if grayscale:
					if image.mode != 'L':
						image = image.convert('L')
				else:
					if image.mode != 'RGB':
						image = image.convert('RGB')
				# Crop it according to top-left corner
				tl_idx = np.random.randint(0, len(topleft)//2)
				row, col = topleft[2*tl_idx:2*tl_idx+2]
				if all([image.size[i] >= self.sizes[group_idx][i] for i in range(2)]):
					box = (col, row, col+self.sizes[group_idx][0], row+self.sizes[group_idx][1])
					image = image.crop(box=box)
				# Eventually resize the image according to target size
				if target_size is not None:
					width_height_tuple = (target_size[1], target_size[0])
					if image.size != width_height_tuple:
						if interpolation not in _PIL_INTERPOLATION_METHODS:
							raise ValueError('Invalid interpolation method {} specified. Supported '
								'methods are {}'.format(interpolation,", ".join(_PIL_INTERPOLATION_METHODS.keys())))
						resample = _PIL_INTERPOLATION_METHODS[interpolation]
						image = image.resize(width_height_tuple, resample)
				# Finally return the image
				return image
			else:
				# If this is not the correct group, continue increasing k
				k += num_classes
		# If no group has been found, then the index must have been out of range
		raise ValueError("Idx out of range")

	def _get_batches_of_transformed_samples(self, index_array):
		batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
		grayscale = self.color_mode == 'grayscale'
		# build batch of image data
		for i, j in enumerate(index_array):
			img = self._load_img(j,
						 grayscale=grayscale,
						 target_size=self.target_size,
						 interpolation=self.interpolation)
			x = img_to_array(img, data_format=self.data_format)
			x = self.image_data_generator.random_transform(x)
			x = self.image_data_generator.standardize(x)
			batch_x[i] = x
		# optionally save augmented images to disk for debugging purposes
		if self.save_to_dir:
			for i, j in enumerate(index_array):
				img = array_to_img(batch_x[i], self.data_format, scale=True)
				fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
																index=j,
																hash=np.random.randint(1e7),
																format=self.save_format)
				img.save(os.path.join(self.save_to_dir, fname))
		# build batch of labels
		if self.class_mode == 'input':
			batch_y = batch_x.copy()
		elif self.class_mode == 'sparse':
			batch_y = index_array
		elif self.class_mode == 'binary':
			batch_y = index_array.astype(K.floatx())
		elif self.class_mode == 'categorical':
			batch_y = np.zeros((len(batch_x), self.num_classes), dtype=K.floatx())
			for i, label in enumerate(index_array):
				batch_y[i, label] = 1.
		else:
			return batch_x
		return batch_x, batch_y

	def next(self):
		"""For python 2.x and called by __next__.

		# Returns
			The next batch.
		"""
		if self.total_batches_seen == self.__len__():
			raise StopIteration
		with self.lock:
			index_array = next(self.index_generator)
		# The transformation of images is not under thread lock
		# so it can be done in parallel
		return self._get_batches_of_transformed_samples(index_array)
		
if __name__ == '__main__':
	import time
	import argparse
	
	parser = argparse.ArgumentParser(description="Attempting and timing database read")
	parser.add_argument("database", help="Input database file path")
	args = vars(parser.parse_args())
	
	target_size = (227,227)
	K.set_image_data_format('channels_first')
	datagen = H5DataGen()
	idg_args = {
		'target_size': target_size,
		'color_mode':'rgb',
		'class_mode': 'categorical',
		'batch_size': 256,
		'shuffle': True
	}
	data_provider = datagen.flow_from_h5file(args["database"], **idg_args)
	print("Length", len(data_provider))
	start = time.time()
	for k, (batch_x, batch_y) in enumerate(data_provider):
		end = time.time()
		print("time", end-start, "\t\tstep", k, "\t\tx:", batch_x.shape, "\ty:", batch_y.shape)
		start = time.time()
