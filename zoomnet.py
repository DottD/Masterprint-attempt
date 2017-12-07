import sys
import os
import psutil
import math
import tensorflow as tf
tfgan = tf.contrib.gan
slim = tf.contrib.slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.python.ops import variable_scope
from feed_images import feed_images


def ZoomBottleNeck(inputs,
					depth,
					block_no=1,
					block_depth=1):
	"""
	Create a ZoomBottleNeck made up by:
		- upsampling that doubles the image size
		- ResNet bottleneck that enhances the image
		
	Args:
		- inputs: 4D tensor [batch size, image width, image height, image depth]
		- depth: final number of classes of the bottleneck
		- block_no: number of the block inside the main program; only used for labelling
		- rep_no: number of the current iteration; only used for labelling
		
	Returns:
		a 4D tensor [batch size, 2x image width, 2x image height, image depth]
	"""
	# Build up the scope
	block_scope = 'Block'+str(block_no)
	# Zoom in layer
	# TODO: Should check for image size
	img_size = tf.shape(inputs)[1]*2
	outputs = tf.image.resize_bilinear(inputs, [img_size, img_size], name=(block_scope+'resize'))
	# ResNet bottlenecks
	for n in range(block_depth):
		resnet_v1.bottleneck(outputs,
							depth,
							2*depth,#TODO: check optimal depth_bottleneck value
							stride=1,
							scope=(block_scope++'/Rep'+str(rep_no)+'_bottleneck'))
	return outputs

def ZoomNet(
	input_length,
	input_batch_size,
	output_img_size,
	output_img_depth,
	max_bottlenecks_block=5,
	scope='ZoomNet'):
	# Compute parameters
	L2 = input_length
	L = math.ceil(math.sqrt(L2))
	if L**2 != L2:
		# If L2 is not a perfect square, change it
		print('Input length is not a perfect square!')
		L2 = L**2
	lexp = math.ceil(math.log2(L))
	l = 2 ** lexp
	Oexp = math.ceil(math.log2(output_img_size))+1 # computed output image size
	O = 2 ** Oexp
	N = Oexp - lexp
	C = lambda k: output_img_depth * ( 4**(N-k-1) )
	Nb = max_bottlenecks_block
	R = lambda k: math.ceil( Nb * math.exp( -0.5 * (2.0*k/(N-1.0)-1.0)**2 ) )
	# Define the function to return
	def net(inputs):
		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			# Fully connected layer
			outputs = slim.fully_connected(inputs, L2, scope='preprocessing')
			# Reshape
			outputs = tf.reshape(outputs, [input_batch_size, L, L, 1], name='reshape')
			# Resize to l x l images
			outputs = tf.image.resize_bilinear(outputs, [l, l], name='initial_resize')
			# ZoomNet blocks
			for n in range(N): # 0,...,N-1
				outputs = ZoomBottleNeck(outputs,
										C(n),
										block_no=n,
										rep_no=R(n))
			# Final resize to fit the requested size
			outputs = tf.image.resize_bilinear(outputs, [output_img_size, output_img_size], name='final_resize')
		return outputs
	return net