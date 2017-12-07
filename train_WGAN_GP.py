#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:07:48 2017

@author: MacD
"""

import sys
import os
import psutil
import math
import tensorflow as tf
tfgan = tf.contrib.gan
slim = tf.contrib.slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.python.ops import variable_scope
from feed_images import feed_images
import zoomnet


tf.flags.DEFINE_integer("img_size", 128, "Expected image side")
tf.flags.DEFINE_integer("img_depth", 1, "Exptected number of channels per image")
tf.flags.DEFINE_integer("batch_size", 128, "Number of images to feed per iteration")
tf.flags.DEFINE_integer("max_iterations", 1000, "Maximum number of iterations")
tf.flags.DEFINE_integer("summary_steps", 1, "Summary every this many steps")
tf.flags.DEFINE_integer("save_steps", 10, "Save checkpoint every this many steps")
tf.flags.DEFINE_float("learning_rate", 0.00005, "Learning rate")
tf.flags.DEFINE_integer("gen_input_size", 100, "Amount of random numbers to feed the generator")
tf.flags.DEFINE_float("gradient_penalty_weight", 10.0, "Coefficient for gradient penalty")

F = tf.flags.FLAGS

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

# Set up generator and discriminator
def generator(inputs):
    net_fn = zoomnet.ZoomNet(F.gen_input_size,
                            F.batch_size,
                            F.img_size,
                            F.img_depth,
                            max_bottlenecks_block=7,
                            scope='Generator')
    return net_fn(inputs)

def discriminator(inputs, gen_inputs):
    disc, _ = resnet_v1.resnet_v1_50(inputs,
                                  num_classes=1,
                                  is_training=False,
                                  global_pool=True,
                                  output_stride=None,
                                  reuse=tf.AUTO_REUSE,
                                  scope='Discriminator')
    return disc

def feed_data():
    # Set up the operations to produce the generator inputs
    noise = tf.random_uniform([F.batch_size, F.gen_input_size*F.img_depth], -1, 1)

    # Set up the operations to feed the GAN with images
    image = feed_images(db_path, 
                        patterns='*.png',
                        ground_truth = False,
                        image_size = F.img_size,
                        image_depth = F.img_depth,
                        batch_size = F.batch_size)
                                      
    return noise, image

def main(_):
    # Adjust the gen_input_size to be a perfect square
    F.gen_input_size = math.ceil(math.sqrt(F.gen_input_size))**2
    # Reset Graph
    tf.reset_default_graph()
    with tf.Graph().as_default():
        # Set up the tfgan estimator
        estimator = tfgan.estimator.GANEstimator(
            generator_fn=generator,
            discriminator_fn=discriminator,
            generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
            discriminator_loss_fn=tfgan.losses.wasserstein_gradient_penalty,
            generator_optimizer=tf.train.AdamOptimizer(F.learning_rate, 0.5),
            discriminator_optimizer=tf.train.AdamOptimizer(F.learning_rate, 0.5),
            add_summaries=[tfgan.estimator.SummaryType.IMAGES,tfgan.estimator.SummaryType.VARIABLES],
            use_loss_summaries=True,
            model_dir=log_dir,
            config=tf.contrib.learn.RunConfig(
                save_summary_steps=F.summary_steps,
                save_checkpoints_steps=F.save_steps
            )
        )

    # Start a new session
    with tf.Session() as sess:
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Variable initialization
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
            
        # Set up and perform the training
        print('Training for ' + str(F.max_iterations) + ' more steps')
        try:
            estimator.train(
                feed_data,
                steps=F.max_iterations
            )
        except KeyboardInterrupt:
            print("Closing session...")
        finally: 
            coord.request_stop()
            
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()