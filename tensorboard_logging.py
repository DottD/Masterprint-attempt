#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""Simple example on how to log scalars and images to tensorboard without tensor ops.
URL: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
License: Copyleft
"""
__author__ = "Michael Gygli"

import os
# import tensorflow as tf
import tensorflow.compat.v1 as tf
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import base64

class Logger(object):
	"""Logging in tensorboard without tensorflow ops."""

	def __init__(self, log_dir):
		"""Creates a summary writer logging to log_dir."""
		self.writer = tf.summary.FileWriter(log_dir)

	def log_scalar(self, tag, value, step):
		"""Log a scalar variable.

		Parameter
		----------
		tag : basestring
			Name of the scalar
		value
		step : int
			training iteration
		"""
		summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
													 simple_value=value)])
		self.writer.add_summary(summary, step)
#		self.writer.flush()

	def log_images(self, tag, images, step):
		"""Logs a list of images."""

		im_summaries = []
		for nr, img in enumerate(images):
			# Write the image to a string
			s = BytesIO()
			plt.imsave(s, img, format='png')
			#imstr = base64.encodebytes(s.getvalue()).decode()

			# Create an Image object
			img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
									   height=img.shape[0],
									   width=img.shape[1])
			# Create a Summary value
			im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
												 image=img_sum))

		# Create and write Summary
		summary = tf.Summary(value=im_summaries)
		self.writer.add_summary(summary, step)
#		self.writer.flush()
		

	def log_histogram(self, tag, values, step, bins=1000, keep=None):
		"""Logs the histogram of a list/vector of values.
		Args:
			- keep: float in [0,100] indicating the percentage of elements to keep (remove outlier)
					(None, default, means no outlier removal)
		"""
		# Convert to a numpy array
		values = np.array(values)
		
		# Remove outliers
		if keep is not None:
			distance = np.abs(values-values.mean())
			t = np.percentile(distance, keep)
			values = values[distance <= t]
		
		# Create histogram using numpy        
		counts, bin_edges = np.histogram(values, bins=bins)

		# Fill fields of histogram proto
		hist = tf.HistogramProto()
		hist.min = float(np.min(values))
		hist.max = float(np.max(values))
		hist.num = int(np.prod(values.shape))
		hist.sum = float(np.sum(values))
		hist.sum_squares = float(np.sum(values**2))

		# Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
		# See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
		# Thus, we drop the start of the first bin
		bin_edges = bin_edges[1:]

		# Add bin edges and counts
		for edge in bin_edges:
			hist.bucket_limit.append(edge)
		for c in counts:
			hist.bucket.append(c)

		# Create and write Summary
		summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
		self.writer.add_summary(summary, step)
#		self.writer.flush()
		
	def log_text(self, tag, text, step=None):
		text_tensor = tf.make_tensor_proto(text, dtype=tf.string)
		meta = tf.SummaryMetadata()
		meta.plugin_data.plugin_name = "text" # ???
		summary = tf.Summary()
		summary.value.add(tag=tag, metadata=meta, tensor=text_tensor)
		self.writer.add_summary(summary, step)
#		self.writer.flush()
		
	def copyFrom(self, path, max_step=None):
		"""
		Copy the content of the given events file.
		"""
		for e in tf.train.summary_iterator(path):
			if not max_step or max_step > e.step:
				self.writer.add_summary(e.summary, e.step)
#				self.writer.flush()
			else:
				return
		
if __name__ == '__main__':
	log_dir = "/Users/MacD/Downloads/classifier_training/net-1-1-1-2  if=1  lr=0.0002  dr=0  bs=256"
	if not os.path.exists(log_dir): os.makedirs(log_dir)
	logger = Logger(log_dir)
	logger.log_text("Description2/bla", "this is a different description", 3)