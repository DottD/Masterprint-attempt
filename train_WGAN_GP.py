#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:07:48 2017

@author: MacD
"""

import sys
import argparse
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
from tensorflow.python.framework import ops
from tensorflow.python.ops.losses import losses
from tensorflow.contrib.gan.python.losses import combine_adversarial_loss
import zoomnet


# Parse command line arguments
parser = argparse.ArgumentParser(description="Train a Wasserstein GAN with gradient penalty to generate fingerprints")
parser.add_argument("--in", help="Full path to the input database directory")
parser.add_argument("--out", help="Name of the output folder, created in the current directory")
parser.add_argument("-s", "--steps",
        default=1000,
        type=int,
        help="Number of training steps")
parser.add_argument("--batch-size", default=128, type=int, help="Number of images to feed per iteration")
parser.add_argument("--img-size", default=(128, 128, 1), 
        type=lambda strin: tuple(int(val) for val in strin.split('x')),
        help="Expected image size in the form 'WxHxD', W=width, H=height, D=depth; H is not used so far")
parser.add_argument("-E", "--summary-steps", default=10, type=int,
        help="Summary every this many steps")
parser.add_argument("-F", "--save-steps", default=100, type=int,
        help="Save checkpoint every this many steps")
parser.add_argument("-G", "--gen-input-size", default=100, type=int,
        help="Amount of random numbers to feed the generator")
args = vars(parser.parse_args())

tf.flags.DEFINE_integer("img_size", args["img_size"][0], "Expected image side")
tf.flags.DEFINE_integer("img_depth", args["img_size"][2], "Exptected number of channels per image")
tf.flags.DEFINE_integer("batch_size", args["batch_size"], "Number of images to feed per iteration")
tf.flags.DEFINE_integer("max_iterations", args["steps"], "Maximum number of iterations")
tf.flags.DEFINE_integer("summary_steps", args["summary_steps"], "Summary every this many steps")
tf.flags.DEFINE_integer("save_steps", args["save_steps"], "Save checkpoint every this many steps")
tf.flags.DEFINE_float("learning_rate", 0.00005, "Learning rate")
tf.flags.DEFINE_integer("gen_input_size", args["gen_input_size"], "Amount of random numbers to feed the generator")
tf.flags.DEFINE_float("gradient_penalty_weight", 10.0, "Coefficient for gradient penalty")

F = tf.flags.FLAGS

# I/O Folders
db_path = os.path.normpath(args["in"]) # Path to the database folder
out_folder = args["out"] # Name of the folder with training outputs
log_dir = os.path.join(os.path.dirname(db_path), out_folder)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
print('Logs will be summarized in ' + log_dir)

# Set up generator and discriminator
def generator(inputs):
    net_fn = zoomnet.ZoomNet(F.gen_input_size,
                            F.batch_size,
                            F.img_size,
                            F.img_depth,
                            max_bottlenecks_block=15,
                            scope='Generator')
    outputs = net_fn(inputs)
    
#    trainable_vars = tf.trainable_variables(scope='Generator')
#    tot_parameters = 0
#    with open('generator.txt', 'w') as f:
#        for var in trainable_vars:
#            tot_parameters += var.get_shape().num_elements()
#            f.write(str(var.name)+'\n')
#    print('Number of generator\'s trainable variables: ' + 
#        str(len(trainable_vars)) + ' with ' + str(tot_parameters) + ' total parameters')
        
    return outputs

def discriminator(inputs, gen_inputs):
    disc, _ = resnet_v1.resnet_v1_50(inputs,
                                  num_classes=1,
                                  is_training=False,
                                  global_pool=True,
                                  output_stride=None,
                                  reuse=tf.AUTO_REUSE,
                                  scope='Discriminator')
    
#    trainable_vars = tf.trainable_variables(scope='Discriminator')
#    tot_parameters = 0
#    with open('discriminator.txt', 'w') as f:
#        for var in trainable_vars:
#            tot_parameters += var.get_shape().num_elements()
#            f.write(str(var.name)+'\n')
#    print('Number of discriminator\'s trainable variables: ' + 
#        str(len(trainable_vars)) + ' with ' + str(tot_parameters) + ' total parameters')

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
    
def disc_loss(gan_model, **kwargs):
    dis_loss = tf.contrib.gan.losses.wasserstein_discriminator_loss(gan_model, add_summaries=True)
    gp_loss = tf.contrib.gan.losses.wasserstein_gradient_penalty(gan_model, add_summaries=True)
    dis_loss += F.gradient_penalty_weight * gp_loss
    return dis_loss

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
            discriminator_loss_fn=disc_loss,
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
            
        # Set up and perform the training
        print('Training for ' + str(F.max_iterations) + ' steps')
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