"""
This file contains the definition of a function that loads the trained 
WGAN-GP model from a checkpoint and then test it, to check the accuracy.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import tensorflow as tf

def plot(gen, n_images, im_dim, out_path, idx):
	# Compute how many rows and cols of images should visualize
	rows = math.floor(math.sqrt(n_images))
	cols = math.ceil(math.sqrt(n_images))
	# Generate images
	z_mb = sample_z(rows*cols, z_dim)
	samples = sess.run(gen, feed_dict={z: z_mb})
	# Create figure and plot sample images
	fig = plt.figure(figsize=(rows, cols))
	gs = gridspec.GridSpec(rows, cols)
	gs.update(wspace=0.05, hspace=0.05)
	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(im_dim[0], im_dim[1]), cmap='Greys_r')
	
	# Save image to file
	file_name = os.path.join(out_path,'{}.png').format(str(idx).zfill(3))
	plt.savefig(file_name, bbox_inches='tight')
	plt.close(fig)
	return idx+1

def sample_z(m, n):
	return np.random.uniform(-1., 1., size=[m, n])

def main(_):
	gen = tf.get_variable("Generator")
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, os.path.normpath(sys.argv[1]))
		print("Checkpoint restored")
