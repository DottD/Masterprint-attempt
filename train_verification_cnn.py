import argparse
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1
from produceTFRecord import input_pipeline
slim = tf.contrib.slim


def create_model(
		iterator,
		tot_images,
		num_subjects=1000,
		img_size=128,
		img_depth=1,
		decay_steps=10,
		reg_coeff=1.0):
	"""
	Create a ResNet-50 model, with the last layer as
	a sigmoid activation function. Batches of images 
	are loaded from the folder provided as "path" argument.
	
	Args:
		- iterator: iterator to the database, see tf.data.
		- op_type: can be either "train" or "eval"; in the former case,
		the output is a training operation, in the latter one it is a
		tuple with loss and number of correct answer.
		- tot_images: total number of images in the database pointed by iterator.
		- num_subjects: number of total output classes.
		- img_size: expected size of each image; if the loaded images are not
		compliant they are scaled to the requested size.
		- img_depth: expected depth of images; images will not be adapted to this
		parameter, so exception or errors may be thrown is it not as expected.
		- decay_steps: number of epochs between two consecutive learning rate reductions.
		
	Returns:
		See op_type argument documentation.
	"""
	# Read images and their correct classification
	images, labels = iterator.get_next()
	labels = tf.one_hot(labels, num_subjects)
	images = tf.image.resize_images(images, [img_size, img_size]) # mostly useless
	images.set_shape([None, img_size, img_size, img_depth])
	# Set up the ResNet architecture
	predictions, _ = resnet_v1.resnet_v1_50(images,
									num_classes=num_subjects,
									is_training=True,
									global_pool=True,
									output_stride=None,
									reuse=tf.AUTO_REUSE,
									scope='Classifier')
	# Actual batch size (may be different at the end of the input queue)
	bs = tf.shape(images)[0]
	# Top layer is a sigmoid activation function, as requested in the paper
	predictions = tf.sigmoid(predictions, name="Sigmoid")
	predictions = tf.reshape(predictions, [bs, -1])
	# Define the loss function
	loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions, labels=labels)
	loss = tf.reduce_sum(loss, axis=1) # Sum along the classes
	loss = tf.reduce_mean(loss) # Mean along the samples
	# Set up regularizers
	reg = tf.contrib.layers.l2_regularizer(scale=reg_coeff/(2*tf.to_float(bs)))
	weights = tf.contrib.framework.get_variables(scope='Classifier') # TODO: filter only weights
	reg_term = tf.contrib.layers.apply_regularization(reg, weights)
	loss += reg_term
	
	### This block should executed only to compute train_op
	# Define a global step counter
	step = tf.train.get_or_create_global_step()
	with tf.variable_scope('time', reuse=tf.AUTO_REUSE):
		epoch = tf.get_variable("epoch", shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0))
		epoch_f = tf.get_variable("epoch_f", shape=[], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
		value = tf.to_float(bs) / tf.to_float(tot_images)
		epoch_f = tf.assign_add(epoch_f, value)
		epoch = tf.to_int32(tf.floor(epoch_f))
	# Define an exponential decay for the learning rate
	learning_rate = tf.train.natural_exp_decay(
		args["learning_rate"],
		epoch,
		decay_steps=decay_steps,
		decay_rate=1/decay_steps
	)
	# Training operation
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam')
	train_op = slim.learning.create_train_op(loss, optimizer, global_step=step)
	
	### This block should be executed only to compute loss and hits
	# Convert labels form one_hot vectors to indices
	labels = tf.argmax(labels, axis=1)
	# Check predictions
	hits = tf.nn.in_top_k(predictions, labels, 1)
	
	return train_op, loss, hits, epoch


if __name__ == '__main__':
	# Start time counter
	startTime = datetime.now()
	print('Computation started at {}'.format(startTime))
	# Fixed parameters
	ckpt_filename = "fp_auth_model.ckpt"
	# Parse command line arguments
	parser = argparse.ArgumentParser(description="Train a ResNet to classify images by subject")
	parser.add_argument("--in", help="Full path to the input database directory")
	parser.add_argument("--out", help="Name of the output folder, created in the current directory")
	parser.add_argument("--load", help="Name of the folder containing the pre-trained model")
	parser.add_argument("-E", "--epochs", default=40, type=int, help="Number of training steps")
	parser.add_argument("--batch-size", default=128, type=int, help="Number of images to feed per iteration")
	parser.add_argument("--img-size", default=(128, 128, 1), 
		type=lambda strin: tuple(int(val) for val in strin.split('x')),
		help="Expected image size in the form 'WxHxD', W=width, H=height, D=depth; H is not used so far")
	parser.add_argument("-S", "--summary-epochs", default=1, type=int, help="Summary every this many epochs")
	parser.add_argument("-D", "--decay-steps", default=10, type=int, help="Epochs between consecutive learning rate reductions")
	parser.add_argument("-R", "--reg-coeff", default=1e-3, type=float, help="Regularization coefficient")
	parser.add_argument("-F", "--save-epochs", default=10, type=int, help="Save checkpoint every this many epochs")
	parser.add_argument("-N", "--num-subjects", default=1000, type=int, help="Total number of people enrolled in the dataset")
	parser.add_argument("-L", "--learning-rate", default=2e-4, type=float, help="Learning Rate")
	args = vars(parser.parse_args())
	print('------')
	print("Parameters:")
	for (key, val) in args.items():
		print(key, '=', val)
	print('------')
	
	# I/O Folders
	db_path = os.path.normpath(args["in"]) # Path to the database folder
	for filename in os.listdir(db_path):
		_, ext = os.path.splitext(filename)
		basename = os.path.basename(filename)
		if ext == '.tfrecords' and basename.startswith('training'):
			train_db_path = os.path.join(db_path, filename)
		elif ext == '.tfrecords' and basename.startswith('validation'):
			valid_db_path = os.path.join(db_path, filename)
	if not 'train_db_path' in globals() or not 'valid_db_path' in globals():
		print('The given folder does not have a training and a validation database in TFRecord format.')
		sys.exit(0)
	out_folder = args["out"] # Name of the folder with training outputs
	load_dir = args["load"]
	log_dir = os.path.join(os.path.dirname(db_path), out_folder)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	print('Logs will be summarized in ' + log_dir)
		
	# Create the input pipeline
	iterator, filename, tot_images = input_pipeline(args["batch_size"])
		
	# Define the model for training
	train_op, loss, hits, epoch_tensor = create_model(
			iterator,
			tot_images,
			num_subjects=args["num_subjects"],
			img_size=args["img_size"][0],
			img_depth=args["img_size"][2],
			decay_steps=args["decay_steps"],
			reg_coeff=args["reg_coeff"]
		)
	
	# Brief check up on the number of variables
	glob_vars = tf.global_variables()
	glob_elems = 0
	with open(os.path.join(log_dir,'variables.txt'), 'w') as f:
		for var in glob_vars:
			glob_elems += var.get_shape().num_elements()
			f.write(str(var.name)+'\n')
	print('Number of CNN\'s variables: ' + str(len(glob_vars)) + ' with ' + str(glob_elems) + ' total elements')
	
	# Initialize a Saver instance
	saver = tf.train.Saver(
		var_list=glob_vars,
		filename=ckpt_filename
	)
	latest_filename = ckpt_filename
			
	# Start a new session
	with tf.Session() as sess:
		# Initialize a Summary writer
		summary_writer_t = tf.summary.FileWriter(os.path.join(log_dir, 'training'), sess.graph)
		summary_writer_v = tf.summary.FileWriter(os.path.join(log_dir, 'validation'), sess.graph)
		
		# Initialize all global variables
		tf.variables_initializer(glob_vars).run()
		tf.local_variables_initializer().run()
		# If a checkpoint is found, restore what you can
		if load_dir is not None:
			ckpt = tf.train.get_checkpoint_state(load_dir, ckpt_filename)
			if ckpt and ckpt.model_checkpoint_path:
				print("Checkpoint found!")
				saver.restore(sess, ckpt.model_checkpoint_path)
				print("Restored!")
			else:
				print("No checkpoint found!")
		else:
			print("Clean execution, nothing to restore!")
			
		# Training procedure
		should_end = False
		epoch = 0
		print("Summary every", args["summary_epochs"], "epochs, saving every", args["save_epochs"], "epochs")
		# Interrupt if the user requests it or if the limit number of epochs is reached
		while not should_end and epoch < args["epochs"]:
			# Training operation (one epoch at a time)
			print("Started training on epoch " + str(epoch) + "...")
			sess.run(iterator.initializer, feed_dict={filename: train_db_path})
			try:
				while True:
					# Feeding mandatory because it is used to create labels
					_, new_epoch = sess.run([train_op, epoch_tensor], feed_dict={filename: train_db_path})
			except KeyboardInterrupt:
				# User terminated with Ctrl-C
				should_end = True
				print('Interrupted during epoch ' + str(epoch))
			except tf.errors.OutOfRangeError:
				# Epoch is finished
				print('Epoch ' + str(epoch) + ' finished')
				
			# Save the model every __ epochs or if the program is about to terminate
			if epoch % args["save_epochs"] == 0 or should_end:
				latest_filename = saver.save(
					sess,
					save_path=os.path.join(log_dir, ckpt_filename),
					global_step=epoch
				)
				print('Model saved on epoch ' + str(epoch) + ' to ' + latest_filename)
				
			# Evaluate and store summaries every __ epochs
			if epoch % args["summary_epochs"] == 0:
				print("Started evaluation on epoch " + str(epoch) + "...")
				
				# Evaluation on training set
				sess.run(iterator.initializer, feed_dict={filename: train_db_path})
				curr_train_loss, curr_train_acc, curr_train_imgs = 0., 0., 0.
				try:	
					while True:
						loc_train_loss, loc_train_hits = sess.run([loss, hits], feed_dict={filename: train_db_path})
						curr_train_loss += loc_train_loss
						curr_train_imgs += len(loc_train_hits)
						curr_train_acc += sum(loc_train_hits)
				except KeyboardInterrupt:
					# User terminated with Ctrl-C
					should_end = True
					print('Evaluation on training set interrupted on epoch ' + str(epoch))
				except tf.errors.OutOfRangeError:
					print('Epoch ' + str(epoch) + ' evaluated on training set')
					# Store summaries
					curr_train_loss /= curr_train_imgs
					curr_train_acc /= curr_train_imgs
					summary_values_t = [
						tf.Summary.Value(tag="Evaluation/loss", simple_value=curr_train_loss),
						tf.Summary.Value(tag="Evaluation/accuracy", simple_value=curr_train_acc*100.0)]
					summary_t = tf.Summary(value=summary_values_t)
					summary_writer_t.add_summary(summary_t, global_step=epoch)
					print('Summary for training written on epoch ' + str(epoch))
					
				# Evaluation on validation set
				sess.run(iterator.initializer, feed_dict={filename: valid_db_path})
				curr_valid_loss, curr_valid_acc, curr_valid_imgs = 0., 0., 0.
				try:
					while True:
						loc_valid_loss, loc_valid_hits = sess.run([loss, hits], feed_dict={filename: valid_db_path})
						curr_valid_loss += loc_valid_loss
						curr_valid_imgs += len(loc_valid_hits)
						curr_valid_acc += sum(loc_valid_hits)
				except KeyboardInterrupt:
					# User terminated with Ctrl-C
					should_end = True
					print('Evaluation on validation set interrupted on epoch ' + str(epoch))
				except tf.errors.OutOfRangeError:
					print('Epoch ' + str(epoch) + ' evaluated on validation set')
					# Store summaries
					curr_valid_loss /= curr_valid_imgs
					curr_valid_acc /= curr_valid_imgs
					summary_values_v = [
						tf.Summary.Value(tag="Evaluation/loss", simple_value=curr_valid_loss),
						tf.Summary.Value(tag="Evaluation/accuracy", simple_value=curr_valid_acc*100.0)]
					summary_v = tf.Summary(value=summary_values_v)
					summary_writer_v.add_summary(summary_v, global_step=epoch)
					print('Summary for validation written on epoch ' + str(epoch))
					
			# Update epoch
			epoch = new_epoch
		
	# Time elapsed
	timeElapsed = datetime.now()-startTime
	print('Time elapsed {}'.format(timeElapsed))
			