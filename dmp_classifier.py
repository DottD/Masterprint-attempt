import os
import argparse
import progressbar
from datetime import datetime
import numpy
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras_contrib.applications.resnet import ResNet
from nist_data_provider import NistDataProvider, to_smooth_categorical
from tensorboard_logging import Logger
	
if __name__ == '__main__':
	# Parse command line arguments
	parser = argparse.ArgumentParser(description="Train a classifier for fingerprints")
	parser.add_argument("in", help="Full path to the input database directory")
	parser.add_argument("out", help="Name of the output folder, created in the current directory")
	parser.add_argument("--load", default=None, help="Name of the folder containing the pre-trained model")
	parser.add_argument("-E", "--epochs", default=600, type=int, help="Number of training steps")
	parser.add_argument("--batch-size", default=64, type=int, help="Number of images to feed per iteration")
	parser.add_argument("--img-size", default=(128, 128, 1), 
		type=lambda strin: tuple(int(val) for val in strin.split('x')),
		help="Expected image size in the form 'WxHxD', W=width, H=height, D=depth; H is not used so far")
	parser.add_argument("-S", "--summary-epochs", default=1, type=int, help="Summary every this many epochs")
	parser.add_argument("--save-epochs", default=1, type=int, help="Save checkpoint every this many epochs")
	parser.add_argument("--learning-rate", default=5E-5, type=float, help="Learning rate for Adam optimizer")
	args = vars(parser.parse_args())
	print('------')
	print("Parameters:")
	for (key, val) in args.items():
		print(key, '=', val)
	print('------')
	
	# Set parameters from cmd line arguments
	img_shape = args["img_size"]
	img_size = img_shape[0]
	nb_epoch = args["epochs"]
	batch_size = args["batch_size"]
	learning_rate = args["learning_rate"]

	# I/O Folders
	db_path = os.path.abspath(os.path.normpath(args["in"]))
	summary_folder = str(datetime.now().isoformat(sep='_', timespec='seconds')).replace(':', '_').replace('-', '_')
	log_dir = os.path.join(os.path.abspath(args["out"]), summary_folder)
	load_path = args["load"]
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	print('Logs will be summarized in ' + log_dir)
	
	# Load data
	provider = NistDataProvider(path=db_path, batch_size=batch_size, validation=None)
	num_classes = provider.num_classes
			
	# Eventually load pre-trained weights
	if load_path:
		CNN = load_model(load_path)
	else:
		# Create and compile models SHOULD NOT BE SOFTMAX!!
		CNN = ResNet(input_shape=img_shape, classes=num_classes, block='bottleneck', residual_unit='v2', repetitions=[2, 2, 2, 2],
		           initial_filters=8, activation="softmax", include_top=True, input_tensor=None, dropout=0.2,
		           transition_dilation_rate=(1, 1), initial_strides=(2, 2), initial_kernel_size=(7, 7),
		           initial_pooling='max', final_pooling='avg', top='classification')
		CNN.compile(optimizer=Adam(lr=learning_rate, amsgrad=True), 
				loss="binary_crossentropy", # not mutually exclusive classes, independent per-class distributions
				metrics=["categorical_accuracy"]) # only after a masterprint multiple classes can be activated
		
	# Initialize a Summary writer
	logger = Logger(log_dir)
	weights = [y for layer in CNN.layers for x in layer.get_weights() for y in x.flatten().tolist()]
	logger.log_histogram("Initialization/weights", weights, 0)
	logger.log_histogram("Initialization/weights_no_outlier", weights, 0, keep=95)
	
	# Create a file with the model description
	with open(os.path.join(log_dir, 'summary_'+summary_folder+'.txt'), mode='w') as F:
		print2F = lambda s: F.write(s+'\n')
		print2F('------')
		print2F("Parameters:")
		for (key, val) in args.items():
			print2F(str(key)+' = '+str(val))
		print2F('------')
		print2F('Logs will be summarized in ' + log_dir)
		CNN.summary(print_fn=print2F)
	
	# Training
	try:
		for e in range(1, nb_epoch+1):
			# Initialize the progress bar
			pb = progressbar.ProgressBar(widgets=[
					'Epoch '+str(e)+'/'+str(nb_epoch)+' ',
					progressbar.widgets.SimpleProgress(format=u'Batch %(value_s)s/%(max_value_s)s'), ' ',
					progressbar.widgets.Bar(marker=u'\u2588'), ' ',
					progressbar.widgets.Timer(), ' ',
					progressbar.widgets.AdaptiveETA()])
			
			for X, Y in pb(provider):
				# Load the batch of images
				Y = to_smooth_categorical(Y, num_classes)
				# Update the CNN
				CNN.train_on_batch(X, Y)

			# Save model weights (every *** epochs)
			if(e % args["save_epochs"] == 0):
				CNN.save(os.path.join(log_dir, 'save_'+summary_folder+'_'+str(e)+'.h5'), overwrite=True)
				with os.scandir(log_dir) as it:
					for entry in it:
						is_curr_save = entry.name.startswith('save_'+summary_folder) and entry.is_file()
						try:
							is_old_save = float(entry.name.split('_')[-1].split('.')[0]) < e
						except ValueError:
							# String not convertible to float -> not to remove
							is_old_save = False
						finally:
							if is_curr_save and is_old_save:
								os.remove(entry.path)

			# Print epoch summary (every *** epochs)
			if(e % args["summary_epochs"] == 0):
				pb = progressbar.ProgressBar(widgets=[
						' -- Evaluation ',
						progressbar.widgets.SimpleProgress(format=u'Batch %(value_s)s/%(max_value_s)s'), ' ',
						progressbar.widgets.Bar(marker=u'\u2588'), ' ',
						progressbar.widgets.Timer(), ' ',
						progressbar.widgets.AdaptiveETA()])
				# Evaluate the model
				accuracy = 0.0
				loss = 0.0
				for X, Y in pb(provider):
					# Load the batch of images
					Y = to_smooth_categorical(Y, num_classes)
					# Generate prediction
					loc_loss, loc_accuracy = CNN.test_on_batch(X, Y)
					loss += loc_loss
					accuracy += loc_accuracy
				loss /= len(provider)
				accuracy /= len(provider)
				# Write summary to file
				logger.log_scalar("Evaluation/accuracy_%", accuracy*100.0, e)
				logger.log_scalar("Evaluation/loss", loss, e)
				weights = [y for layer in CNN.layers for x in layer.get_weights() for y in x.flatten().tolist()]
				logger.log_histogram("Model/weights", weights, e)
				logger.log_histogram("Model/weights_no_outlier", weights, e, keep=95)
	except KeyboardInterrupt:
		print("The user interrupted the training.")
