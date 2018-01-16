import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.backend import cast_to_floatx
from keras.models import Model
from keras.layers import Input
from keras import initializers
from keras.utils import generic_utils, np_utils
from keras.optimizers import RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Activation, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import AveragePooling2D
import matplotlib.pylab as plt
from PIL import Image
import time
import math
import os
import argparse
import progressbar
# Files
import nistdata
import utils


#DCGAN
def generator(noise_dim, s):
	#noise_dim: Dimension of the noise input
	#s: height of square image output
	#s must be a power of 2
	
	#Variables
	img_dim = (s, s, 1)               #Output Image Shape
	f = 1024                          #Starting Filters
	nb_upconv = 5                     #Number of Deconvs
	start_dim = int(s / 2**nb_upconv) #Start Dimension
	output_channels = img_dim[2]      #RGB or Black&white

	gen_input = Input(shape=(noise_dim,))

	# Noise input and reshaping
	x = Dense(f * start_dim * start_dim, use_bias=False)(gen_input)
	x = Reshape((start_dim, start_dim, f))(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)

	# Transposed conv blocks: Deconv2D->BN->ReLU
	for i in range(nb_upconv - 1):
		nb_filters = int(f / (2 ** (i + 1)))
		x = UpSampling2D(size=(2,2))(x)
		x = Conv2D(nb_filters, kernel_size=(3, 3), padding="same")(x)
		x = BatchNormalization()(x)
		x = Activation("relu")(x)

	# Last block
	x = UpSampling2D(size=(2,2))(x)
	x = Conv2D(output_channels, kernel_size=(3, 3), padding="same")(x)
	x = Activation("tanh")(x)

	generator_model = Model(inputs=[gen_input], outputs=[x])
	#visualize_model(generator_model)

	return generator_model

def discriminator(s):
	#s: width of square image output

	img_dim = (s, s, 1)
	min_s = min(img_dim[:1])
	bn_axis = 1

	disc_input = Input(shape=img_dim)

	# Get the list of number of conv filters
	# (first layer starts with 64), filters are subsequently doubled
	nb_conv = int(np.floor(np.log(min_s // 4) / np.log(2)))
	list_f = [64 * min(8, (2 ** i)) for i in range(nb_conv)]

	# First conv with 2x2 strides
	x = Conv2D(list_f[0], kernel_size=(3, 3), strides=(2, 2),
				padding="same", use_bias=False, 
				kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None))(disc_input)
	x = BatchNormalization(axis=bn_axis)(x)
	x = LeakyReLU(0.2)(x)

	# Conv blocks: Conv2D(2x2 strides)->BN->LReLU
	for i, f in enumerate(list_f[1:]):
		x = Conv2D(f, 
				kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False,
				kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None))(x)
		x = BatchNormalization(axis=bn_axis)(x)
		x = LeakyReLU(0.2)(x)

	# Last convolution
	x = Conv2D(1, 
			kernel_size=(3, 3), padding="same", use_bias=False,
			kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None))(x)
	# Average pooling
	x = AveragePooling2D(pool_size=(4, 4))(x)
	x = Flatten()(x)

	discriminator_model = Model(inputs=[disc_input], outputs=[x])

	return discriminator_model

def dcgan(generator, discriminator, noise_dim):
	#generator: keras generator model
	#discriminator: keras discriminator model
	#noise_dim: generator input noise dimension

	noise_input = Input(shape=(noise_dim,))
	generated_image = generator(noise_input)
	DCGAN_output = discriminator(generated_image)

	DCGAN = Model(inputs=[noise_input], outputs=[DCGAN_output])

	return DCGAN
	
if __name__ == '__main__':
	# Fixed parameters
	ckpt_filename = "fp_auth_model.ckpt"
	# Parse command line arguments
	parser = argparse.ArgumentParser(description="Train a WGAN to generate fingerprints")
	parser.add_argument("in", help="Full path to the input database directory")
	parser.add_argument("out", help="Name of the output folder, created in the current directory")
	parser.add_argument("--load", help="Name of the folder containing the pre-trained model")
	parser.add_argument("-E", "--epochs", default=600, type=int, help="Number of training steps")
	parser.add_argument("--batch-size", default=64, type=int, help="Number of images to feed per iteration")
	parser.add_argument("--img-size", default=(128, 128, 1), 
		type=lambda strin: tuple(int(val) for val in strin.split('x')),
		help="Expected image size in the form 'WxHxD', W=width, H=height, D=depth; H is not used so far")
	parser.add_argument("-S", "--summary-epochs", default=1, type=int, help="Summary every this many epochs")
	parser.add_argument("--save-epochs", default=1, type=int, help="Save checkpoint every this many epochs")
	parser.add_argument("--noise-length", default=100, type=int, help="Total number of people enrolled in the dataset")
	parser.add_argument("--learning-rate", default=5e-5, type=float, help="Learning Rate")
	parser.add_argument("--disc-train-steps", default=5, type=int, help="Number of iterations of the critic per generator iteration")
	args = vars(parser.parse_args())
	print('------')
	print("Parameters:")
	for (key, val) in args.items():
		print(key, '=', val)
	print('------')
	# Set parameters from cmd line arguments
	noise_dim = args["noise_length"]
	img_size = args["img_size"][0]
	learning_rate = args["learning_rate"]
	nb_epoch = args["epochs"]
	batch_size = args["batch_size"]
	discriminator_iterations = args["disc_train_steps"]
	clamp_upper = 0.01
	clamp_lower = -0.01
	noise_scale = 0.5

	# I/O Folders
	db_path = os.path.abspath(os.path.normpath(args["in"])) # Path to the database folder
	out_folder = args["out"] # Name of the folder with training outputs
	load_dir = args["load"]
	log_dir = os.path.abspath(os.path.join(os.path.dirname(db_path), out_folder))
	output_log_file = os.path.join(log_dir, 'output.txt')
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	print('Logs will be summarized in ' + log_dir)
		
	#Input
	def sample_noise(noise_dim, batch_size, noise_scale):
		return np.random.normal(scale=noise_scale, size=(batch_size, noise_dim))
	#Loss
	def wasserstein(y_true, y_pred):
		return K.mean(y_true * y_pred)
	# Create models
	generator_model = generator(noise_dim,img_size)
	discriminator_model = discriminator(img_size)
	dcgan_model = dcgan(generator_model, discriminator_model, noise_dim)
	# Compile models
	rmsprop = RMSprop(lr=learning_rate)
	generator_model.compile(loss='mse', optimizer=rmsprop)
	discriminator_model.trainable = False
	dcgan_model.compile(loss=wasserstein, optimizer=rmsprop)
	discriminator_model.trainable = True
	generator_model.trainable = False
	discriminator_model.compile(loss=wasserstein, optimizer=rmsprop)
	generator_model.trainable = True
	# Eventually load weights
	if load_dir:
		utils.loadModelWeights(generator_model, discriminator_model, load_dir)

	# Load data
	def tanh_transform(x):
		return x*2/255 - 1
	X_train, Y_train = nistdata.load_data(s = 256, dirname = db_path)
	nb_img = X_train.shape[0]
	datagen = nistdata.DataGenerator(crop_size=img_size, preprocessing_function = tanh_transform)
	gen_batch = datagen.flow_random(X = X_train, batch_size = batch_size)
	
	# Initialize a Summary writer
	summary_writer = tf.summary.FileWriter(os.path.join(log_dir, 'summary'))

	# Training
	for e in range(1, nb_epoch+1):
		# Compute the number of batch per epoch
		n_batch_per_epoch = math.ceil(nb_img / batch_size)
		epoch_size = n_batch_per_epoch * batch_size
		# Initialize the progress bar
		pbDiscText = progressbar.widgets.FormatCustomText('Disc %(it)d/%(tot)d', dict(it=1, tot=100))
		pb = progressbar.ProgressBar(widgets=[
				'Epoch '+str(e)+'/'+str(nb_epoch)+' ',
				progressbar.widgets.SimpleProgress(format=u'Batch %(value_s)s/%(max_value_s)s'), ' ',
				pbDiscText, ' ',
				progressbar.widgets.Bar(marker=u'\u2588'), ' ',
				progressbar.widgets.Timer(), ' ',
				progressbar.widgets.AdaptiveETA()])

		for batch_counter in pb(range(n_batch_per_epoch)):
			########## 1) Train the critic / discriminator ############
			list_disc_loss_real = []
			list_disc_loss_gen = []
			# Many critic updates at the beginning
			if (batch_counter < 25 and not load_dir) or batch_counter % 500 == 0:
				disc_iterations = 100
			else:
				disc_iterations = discriminator_iterations
			for disc_it in range(disc_iterations):
				pbDiscText.update_mapping(it=disc_it+1, tot=disc_iterations)
				# Clip discriminator weights
				for l in discriminator_model.layers:
					weights = l.get_weights()
					weights = [np.clip(w, clamp_lower, clamp_upper) for w in weights]
					l.set_weights(weights)
				# Load the batch of images
				X_real_batch = gen_batch.next()
				# Create the fake images
				noise_input = sample_noise(noise_dim, batch_size, noise_scale)
				X_disc_gen = generator_model.predict(noise_input)
				# Update the discriminator
				disc_loss_real = discriminator_model.train_on_batch(X_real_batch, -np.ones(X_real_batch.shape[0]))
				disc_loss_gen = discriminator_model.train_on_batch(X_disc_gen, np.ones(X_disc_gen.shape[0]))
				list_disc_loss_real.append(disc_loss_real)
				list_disc_loss_gen.append(disc_loss_gen)
			
			########## 2) Train the generator ############
			X_gen = sample_noise(noise_dim, batch_size, noise_scale)
			# Freeze the discriminator
			discriminator_model.trainable = False
			gen_loss = dcgan_model.train_on_batch(X_gen, -np.ones(batch_size))
			# Unfreeze the discriminator
			discriminator_model.trainable = True

			# Compute losses
			lossD = -np.mean(list_disc_loss_real) - np.mean(list_disc_loss_gen) #-6*np.mean(list_disc_loss)
			lossG = -gen_loss
			
			# Summary operations every *** epochs
			if(batch_counter==n_batch_per_epoch-1 and e % args["summary_epochs"] == 0):
				with tf.Session() as sess:
					tf.summary.image("Evaluation/fake_images", X_gen, max_outputs=4)
					tf.summary.image("Evaluation/true_images", X_real_batch, max_outputs=4)
					tf.summary.scalar("Evaluation/lossD", lossD)
					tf.summary.scalar("Evaluation/lossG", lossG)
					summary_op = tf.summary.merge_all()
					summary = sess.run(summary_op)
					summary_writer.add_summary(summary, global_step=e)
					summary_writer.flush()
			# Save images for visualization every 5 batches
			if batch_counter % 5 == 0:
				utils.plotGeneratedBatch(X_real_batch, X_disc_gen, os.path.join(log_dir, 'currentSample.png'))

		# Save model weights (every *** epochs)
		if(e % args["save_epochs"] == 0):
			utils.saveModelWeights(generator_model, discriminator_model, log_dir)
