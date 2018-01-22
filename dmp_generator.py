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
from tensorboard_logging import Logger


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

	G = Model(inputs=[gen_input], outputs=[x])

	return G

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

	D = Model(inputs=[disc_input], outputs=[x])

	return D

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
	parser.add_argument("-E", "--epochs", default=2000, type=int, help="Number of training steps")
	parser.add_argument("--batch-size", default=64, type=int, help="Number of images to feed per iteration")
	parser.add_argument("--img-size", default=(128, 128, 1), 
		type=lambda strin: tuple(int(val) for val in strin.split('x')),
		help="Expected image size in the form 'WxHxD', W=width, H=height, D=depth; H is not used so far")
	parser.add_argument("-S", "--summary-epochs", default=1, type=int, help="Summary every this many epochs")
	parser.add_argument("--save-epochs", default=1, type=int, help="Save checkpoint every this many epochs")
	parser.add_argument("--noise-length", default=100, type=int, help="Total number of people enrolled in the dataset")
	parser.add_argument("--learning-rate", default=5e-6, type=float, help="Learning Rate")
	parser.add_argument("--decay-rate", default=0.01, type=float, help="Decay Rate")
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
	decay_rate = args["decay_rate"]
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
		
	# Load data
	def tanh_transform(x):
		return x*2/255 - 1
	X_train, Y_train = nistdata.load_data(s = 256, dirname = db_path)
	nb_img = X_train.shape[0]
	datagen = nistdata.DataGenerator(crop_size=img_size, preprocessing_function = tanh_transform)
	gen_batch = datagen.flow_random(X = X_train, batch_size = batch_size)
	
	#Input
	def sample_noise(noise_dim, batch_size, noise_scale):
		return np.random.normal(scale=noise_scale, size=(batch_size, noise_dim))
	#Loss
	def wasserstein(y_true, y_pred):
		return -K.mean(y_true * y_pred)
	# Useful function
	def set_trainability(model, trainable=False):
		model.trainable = trainable
		for layer in model.layers:
			layer.trainable = trainable
	# Create models
	G = generator(noise_dim,img_size)
	G.compile(loss='mse', optimizer=RMSprop(lr=learning_rate, decay=decay_rate*batch_size/nb_img))
	
	D = discriminator(img_size)
	D.compile(loss=wasserstein, optimizer=RMSprop(lr=learning_rate, decay=decay_rate*batch_size/nb_img))
	
	DCGAN = dcgan(G, D, noise_dim)
	set_trainability(D, False)
	DCGAN.compile(loss=wasserstein, optimizer=G.optimizer)
	
	# Eventually load weights
	if load_dir:
		utils.loadModelWeights(G, D, load_dir)
	
	# Initialize a Summary writer
	logger = Logger(os.path.join(log_dir, 'summary'))

	# Training
	try:
		for e in range(1, nb_epoch+1):
			# Compute the number of batch per epoch
			n_batch_per_epoch = math.ceil(nb_img / batch_size)
			epoch_size = n_batch_per_epoch * batch_size
			# Many critic updates at the beginning
			if (e < 25 and not load_dir) or e % 500 == 0:
				disc_iterations = 100
			else:
				disc_iterations = discriminator_iterations
			# Initialize the progress bar
			pb = progressbar.ProgressBar(max_value=disc_iterations+1,
				widgets=['Epoch '+str(e)+'/'+str(nb_epoch)+' ',
						progressbar.widgets.SimpleProgress(format=u'Steps %(value_s)s/%(max_value_s)s'), ' ',
						progressbar.widgets.Bar(marker=u'\u2588'), ' ',
						progressbar.widgets.Timer(), ' ',
						progressbar.widgets.AdaptiveETA()])

			########## 1) Train the critic / discriminator ############
			# Initialize the labels [Y_fake, Y_real]
			Y_real = np.ones(batch_size)
			Y_fake = -Y_real
			Y = np.concatenate([Y_real, Y_fake], axis=0)
			# Initializations
			lossD = []
			set_trainability(D, True)
			for _ in range(disc_iterations):
				# Clip discriminator weights
				for l in D.layers:
					weights = l.get_weights()
					weights = [np.clip(w, clamp_lower, clamp_upper) for w in weights]
					l.set_weights(weights)
				# Load the batch of images
				X_real = gen_batch.next()
				# Create the fake images
				noise_input = sample_noise(noise_dim, batch_size, noise_scale)
				X_fake = G.predict(noise_input)
				# Create the batch
				X = np.concatenate([X_real, X_fake], axis=0)
				# Update the discriminator
				lossD.append(D.train_on_batch(X, Y))
				# Update progressbar
				pb.update(_)
			set_trainability(D, False)
			# Compute the total loss
			lossD = np.mean(lossD)
			
			########## 2) Train the generator ############
			X = sample_noise(noise_dim, batch_size, noise_scale)
			lossG = DCGAN.train_on_batch(X, Y_real)
			# Update progressbar
			pb.update(disc_iterations+1)
			pb.finish()
				
			# Summary operations every *** epochs
			if e % args["summary_epochs"] == 0:
				n_img = 4
				X = sample_noise(noise_dim, n_img, noise_scale)
				X_fake = G.predict(X)
				X_fake = [np.squeeze(np.array(img)) for img in X_fake.tolist()]
				X_real = [np.squeeze(np.array(img)) for img in X_real[:n_img,:,:,:].tolist()]
				def loss2acc(loss):
					# Loss is in [-1,1] where -1 stands for "perfectly-trained"
					return (-loss + 1.) / 2. * 100.
				logger.log_scalar("Evaluation/lossD", lossD, e)
				logger.log_scalar("Evaluation/lossG", lossG, e)
				logger.log_scalar("Evaluation/accuracyD", loss2acc(lossD), e)
				logger.log_scalar("Evaluation/accuracyG", loss2acc(lossG), e)
				logger.log_images("Evaluation/fake_images", X_fake, e)
				logger.log_images("Evaluation/real_images", X_real, e)
				weights = [y for layer in D.layers for x in layer.get_weights() for y in x.flatten().tolist()]
				logger.log_histogram("Discriminator/weights", weights, e)
				logger.log_histogram("Discriminator/weights_no_outliers", weights, e, keep=95)
				weights = [y for layer in G.layers for x in layer.get_weights() for y in x.flatten().tolist()]
				logger.log_histogram("Generator/weights", weights, e)
				logger.log_histogram("Generator/weights_no_outliers", weights, e, keep=95)
				
			# Save model weights (every *** epochs)
			if e % args["save_epochs"] == 0:
				utils.saveModelWeights(G, D, log_dir)
	
	except KeyboardInterrupt:
		print("The user interrupted the training.")
		