# WGAN taken from https://github.com/ConnorJL/WGAN-Tensorflow
# modified by Filippo Santarelli according to improvements
# by https://arxiv.org/pdf/1704.00028.pdf

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import ops
from load_batch import scan_dir
from load_batch import load_random_batch


db_path = os.path.normpath(sys.argv[1]) # Path to the database folder
out_folder = sys.argv[2] # Name of the folder with training outputs
# Recursively scan the db folder to find .png files
file_list = scan_dir(db_path)
log_dir = os.path.join(os.path.dirname(db_path), out_folder)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
print('Logs will be saved to ' + log_dir)

slim = tf.contrib.slim

img_size = 128 # Not all the code is compatible
img_depth = 1 # Number of channels of each image (Not all the code is compatible)
batch_size = 64 # Must be representable through tf.uint8
max_iterations = 100000000
sum_per = 50 # Create a summary every this many steps
sum_n = 4 # Number of images shown in summary per iteration
save_per = 1000 # Save every this many steps
learning_rate = 0.00005
d_iters = 5 # Number of discriminator training steps per generator training step
z_dim = 100 # Dimension of the noise vector
c = 0.01 # Value to which to clip the discriminator weights
clip_per = 1 # Experimental. Clip discriminator weights every this many steps. Only works reliably if clip_per=<d_iters
gp_lambda = 10 # Gradient penalty lambda correction

#batch_size=1
#d_iters=1
#sum_per =1
#print('Warning: debug enabled!')

def generator(z, training=True, weight_decay=0.0001, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    batch_norm_params = {
        'is_training': training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': ops.GraphKeys.UPDATE_OPS,
    }

    # an affine transformation is applied to each sample by 1024 neurons
    gen = slim.fully_connected(z, 1024)
    # each neuron scalar output is reshaped to a 1x1 matrix
    gen = tf.reshape(gen, [batch_size, 1, 1, 1024])

    with arg_scope(
        [slim.conv2d_transpose],
        weights_regularizer=regularizers.l2_regularizer(weight_decay),
        weights_initializer=initializers.variance_scaling_initializer(),
        activation_fn=nn_ops.relu,
        normalizer_fn=layers.batch_norm,
        normalizer_params=batch_norm_params):

        # Each tuple is (number of channels, kernel size, stride)
        l = [(1024, 3, 2), # 1024 2x2 output matrices
             (512, 3, 2), # 512 4x4 output matrices
             (256, 3, 2), # 256 8x8 output matrices
             (128, 3, 2), # 128 16x16 output matrices
             (128, 3, 2), # 128 32x32 output matrices
             (64, 3, 2), # 64 64x64 output matrices
             (img_depth, 3, 2)] # img_depth 128x128 output matrices
        gen = slim.stack(gen, slim.conv2d_transpose, l, scope="transposed_convolution")

    # add non linearity
    gen = tf.tanh(gen)

    return gen


def discriminator(img, training=True, weight_decay=0.0001, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    batch_norm_params = {
        'is_training': training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': ops.GraphKeys.UPDATE_OPS,
    }

    with arg_scope(
        [slim.conv2d],
        weights_regularizer=regularizers.l2_regularizer(weight_decay),
        weights_initializer=initializers.variance_scaling_initializer(),
        activation_fn=nn_ops.relu,
        normalizer_fn=layers.batch_norm,
        normalizer_params=batch_norm_params):

        # Each tuple is (number of channels, kernel size, stride)
        l = [(64, 3, 2), # 64 64x64 output matrices
             (128, 3, 2), # 128 32x32 output matrices
             (256, 3, 2), # 256 16x16 output matrices
             (512, 3, 2), # 512 8x8 output matrices
             (1024, 3, 2)] # 1024 4x4 output matrices
        disc = slim.stack(img, slim.conv2d, l, scope="convolution")

    # each matrix of the last layer is put in a row
    disc = tf.reshape(disc, [batch_size, 4*4*1024])
    # affine transformation to produce only one output
    disc = slim.fully_connected(disc, 1, activation_fn=None, scope="logits")

    return disc


def main():
    # Feed your data here! The program expects batches of 128x128x3 float32
    # (normalized to be between 0 and 1) images by default
    images = tf.placeholder(tf.float32, [batch_size, img_size, img_size, img_depth], name='images')
    tf.summary.image("real", images, max_outputs=sum_n)

    z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z')

    with tf.variable_scope("generator") as scope:
        gen = generator(z)
        tf.summary.image("fake", gen, max_outputs=sum_n)

    with tf.variable_scope("discriminator") as scope:
        disc_real = discriminator(images)
        scope.reuse_variables()
        disc_fake = discriminator(gen)

    # Define losses as in https://gist.github.com/mjdietzx/a8121604385ce6da251d20d018f9a6d6
    # and in generative-model...
    # Function that computes Earth mover distance of a tensor distribution
    def em_loss(_tensor):
        return tf.reduce_mean(_tensor)
    # Function that computes x_hat (batch_size, ...)
    def rand_mid_point(_x, _x_tilde):
        _epsilon = tf.random_normal([batch_size, 1, 1, 1])
        return _epsilon * _x + (1.0 - _epsilon) * _x_tilde
    # Function that computes the gradient penalty
    def grad_penalty(_D, _x, _x_tilde, _lambda):
        _x_hat = rand_mid_point(_x, _x_tilde)
        _grad = tf.gradients(_D(_x_hat), [_x_hat])[0]
        _grad_norm = tf.sqrt(tf.reduce_sum((_grad)**2, axis=1))
        return _lambda * tf.reduce_mean((_grad_norm - 1)**2)
    # Losses
    disc_real_loss = em_loss(disc_real)
    disc_fake_loss = em_loss(disc_fake)
    d_loss = disc_real_loss - disc_fake_loss + \
        grad_penalty(discriminator, images, gen, gp_lambda)

    # Define Losses
#    disc_real_loss = tf.losses.sigmoid_cross_entropy(tf.ones([batch_size, 1]), disc_real)
#    disc_fake_loss = tf.losses.sigmoid_cross_entropy(tf.fill([batch_size, 1], -1.0), disc_fake)
#
#    d_loss = disc_real_loss + disc_fake_loss
    g_loss = tf.losses.sigmoid_cross_entropy(tf.ones([batch_size, 1]), disc_fake)

    tf.summary.scalar("Discriminator_loss_real", disc_real_loss)
    tf.summary.scalar("Discrimintator_loss_fake", disc_fake_loss)
    tf.summary.scalar("Discriminator_loss", d_loss)
    tf.summary.scalar("Generator_loss", g_loss)

    # The paper found RMSProp to work better than Adam or other momentum based methods
    d_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    g_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
    # Create training ops
    d_train_op = slim.learning.create_train_op(d_loss, d_optimizer, variables_to_train=d_vars)
    g_train_op = slim.learning.create_train_op(g_loss, g_optimizer, variables_to_train=g_vars)

    with tf.Session() as sess:
#        sess = tf_d1ebug.LocalCLIDebugWrapperSession(sess)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        saver = tf.train.Saver()

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        start = 0
        # If a checkpoint is found, restore what you can. If not, continue
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Checkpoint found! Restoring...")
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Hackey way to determine what step we're starting on. It feels like there should be some in built function in TensorFlow to do this but I can't find any...
            start = int(ckpt.model_checkpoint_path.split("-")[-1])+1
            print("Restored!")
        else:
            print("No checkpoint found!")

        try:
            current_step = start
            print("Starting training!")
            for itr in range(start, max_iterations):

                # As per the reference implementation, the discriminator gets a lot of training early on
                if current_step < 25 or current_step % 500 == 0:
                    diters = 100
                else:
                    diters = d_iters

                # Load a batch of images
                batch_img = load_random_batch(file_list, batch_size)

                # Train discriminator several times
                for i in range(diters):
                    batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
                    sess.run(d_train_op, feed_dict={z: batch_z, images: batch_img})

                # Train generator once
                batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
                sess.run(g_train_op, feed_dict={z: batch_z, images: batch_img})


                # Give the user some feedback
                if itr % sum_per == 0:
                    g_loss_val, d_loss_val, summary_str = sess.run([g_loss, d_loss, summary_op], feed_dict={z: batch_z, images: batch_img})
                    print("Step: %d, Generator Loss: %g, Discriminator Loss: %g" % (itr, g_loss_val, d_loss_val))
                    summary_writer.add_summary(summary_str, itr)

                # Every so often save
                if itr % save_per == 0:
                    saver.save(sess, os.path.join(log_dir, "model.ckpt"), global_step=itr)
                current_step = itr

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached!')

        except KeyboardInterrupt:
            print("Ending training...")
            # User terminated with Ctrl-C, save current state
            saver.save(sess, os.path.join(log_dir, "model.ckpt"), global_step=current_step)

        finally:
            coord.request_stop()

        # Done!
        coord.join(threads)


if __name__ == "__main__":
    main()
