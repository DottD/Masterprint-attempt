#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 1 13:55:00 2017

@author: MacD
"""
import sys
import os
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1
from feed_images import feed_images
slim = tf.contrib.slim


tf.flags.DEFINE_integer("num_subjects", 5400, "Total number of people enrolled in the dataset")
tf.flags.DEFINE_integer("img_size", 128, "Expected image side")
tf.flags.DEFINE_integer("img_depth", 1, "Exptected number of channels per image")
tf.flags.DEFINE_integer("batch_size", 64, "Number of images to feed per iteration")
tf.flags.DEFINE_integer("max_iterations", 1000, "Maximum number of iterations")
tf.flags.DEFINE_integer("summary_secs", 60, "Summary every this many seconds")
tf.flags.DEFINE_integer("save_secs", 600, "Save checkpoint every this many seconds")
tf.flags.DEFINE_float("learning_rate", 0.00005, "Learning rate")
tf.flags.DEFINE_integer("gen_input_side", 10, "Side of the square of random numbers to feed the generator")
tf.flags.DEFINE_float("gradient_penalty_weight", 10.0, "Coefficient for gradient penalty")

F = tf.flags.FLAGS

def main(_):
	# I/O Folders
	db_path = os.path.normpath(sys.argv[1]) # Path to the database folder
	out_folder = sys.argv[2] # Name of the folder with training outputs
	log_dir = os.path.join(os.path.dirname(db_path), out_folder)
	save_dir = os.path.join(log_dir, 'ckpt')
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	print('Logs will be summarized in ' + log_dir + ' and saved to ' + save_dir)
	
	# Read images and their correct classification
	images, labels = feed_images(db_path, '*.png', 
						ground_truth=True,
						image_size=F.img_size,
						image_depth=F.img_depth,
						num_classes=F.num_subjects,
						batch_size=F.batch_size)
						
	# Set up the ResNet architecture
	predictions, _ = resnet_v1.resnet_v1_101(images,
									num_classes=F.num_subjects,
									is_training=True,
									global_pool=True,
									output_stride=None,
									reuse=tf.AUTO_REUSE,
									scope='resnet')
	predictions = tf.reshape(predictions, [F.batch_size, -1])
	
	# Define the loss function
	loss = slim.losses.softmax_cross_entropy(predictions, labels)
	loss = tf.Print(loss, [loss*100], message="Loss x100: ")
	tf.summary.scalar("loss", loss)
	
	# Training operation
	optimizer = tf.train.AdamOptimizer(learning_rate=F.learning_rate, name='Adam')
	train_op = slim.learning.create_train_op(loss, optimizer)
		
	# Add ops to restore all the variables.
	restorer = tf.train.Saver()
		
	# Start a new session
	with tf.Session() as sess:
		# If a checkpoint is found, restore what you can. If not, initialize variables and continue
		ckpt = tf.train.get_checkpoint_state(log_dir)
		if ckpt and ckpt.model_checkpoint_path:
			print("Checkpoint found! Restoring...")
			restorer.restore(sess, ckpt.model_checkpoint_path)
			print("Restored!")
		else:
			tf.global_variables_initializer().run()
			print("No checkpoint found!")
			
		# Training	
		slim.learning.train(
			train_op,
			log_dir,
			number_of_steps=F.max_iterations,
			save_summaries_secs=F.summary_secs,
			save_interval_secs=F.save_secs,
			saver=restorer)

if __name__ == '__main__':
		tf.app.run()