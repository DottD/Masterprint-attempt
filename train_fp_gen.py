import argparse
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1
from produceTFRecord import input_pipeline
import zoomnet
slim = tf.contrib.slim
tfgan = tf.contrib.gan


def create_model(
		iterator,
		tot_images,
		noise_length=100,
		batch_size=64,
		img_size=128,
		img_depth=1,
		decay_steps=10,
		grad_pen_coeff=10.0,
		max_bottlenecks_block=6):
	"""
	Create a ResNet-50 model, with the last layer as
	a sigmoid activation function. Batches of images 
	are loaded from the folder provided as "path" argument.
	
	Args:
		...
		
	Returns:
		See op_type argument documentation.
	"""
	def generator(inputs):
		"""
		Define the generator network
		"""
		net_fn = zoomnet.ZoomNet(noise_length,
								batch_size,
								img_size,
								img_depth,
								max_bottlenecks_block,
								scope='Generator')
		outputs = net_fn(inputs)
		return outputs

	def discriminator(inputs):
		"""
		Define the discriminator network
		"""
		disc, _ = resnet_v1.resnet_v1_50(inputs,
										num_classes=1,
										is_training=False,
										global_pool=True,
										output_stride=None,
										reuse=tf.AUTO_REUSE,
										scope='Discriminator')
		return disc
	# Read a batch of images (function definition)
	true_images, _ = iterator.get_next()
	true_images = tf.image.resize_images(true_images, [img_size, img_size]) # mostly useless
	true_images.set_shape([None, img_size, img_size, img_depth])
	# Actual batch size (may be different at the end of the input queue)
	bs = tf.shape(true_images)[0]
	# Set up the architecture
	noise = tf.random_uniform([bs, noise_length*img_depth], -1, 1)
	fake_images = generator(noise)
	fake_disc = discriminator(fake_images)
	true_disc = discriminator(true_images)
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
		return _lambda * (_grad_norm - 1)**2
	# Losses (discriminator yields 1 when recognizes false images)
	d_loss = tf.reduce_mean( true_disc - fake_disc + grad_penalty(discriminator, true_images, fake_images, grad_pen_coeff) )
	g_loss = - tf.reduce_mean( fake_disc )
	
	### This block should executed only to compute the training operations
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
	d_train_op = slim.learning.create_train_op(d_loss, optimizer, global_step=step)
	g_train_op = slim.learning.create_train_op(g_loss, optimizer, global_step=step)
	
	### This block should be executed only to compute the accuracy
	d_accuracy = 1.0 - tf.reduce_mean(true_disc)
	g_accuracy = tf.reduce_mean(fake_disc)
	
	return d_train_op, g_train_op, d_accuracy, g_accuracy, fake_images


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
	parser.add_argument("--decay-steps", default=10, type=int, help="Epochs between consecutive learning rate reductions")
	parser.add_argument("--save-epochs", default=10, type=int, help="Save checkpoint every this many epochs")
	parser.add_argument("--noise-length", default=100, type=int, help="Total number of people enrolled in the dataset")
	parser.add_argument("--learning-rate", default=1e-5, type=float, help="Learning Rate")
	parser.add_argument("--disc-train-steps", default=5, type=int, help="Number of iterations of the critic per generator iteration")
	parser.add_argument("--grad-pen-coeff", default=10.0, type=float, help="Gradient penalty coefficient")
	parser.add_argument("--max-bottlenecks-block", default=10, type=int, help="Maximum block repetition in ZoomNet")
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
	d_train_op, g_train_op, d_accuracy, g_accuracy, fake_images = create_model(
			iterator,
			tot_images,
			noise_length=args["noise_length"],
			batch_size=args["batch_size"],
			img_size=args["img_size"][0],
			img_depth=args["img_size"][2],
			decay_steps=args["decay_steps"],
			grad_pen_coeff=args["grad_pen_coeff"],
			max_bottlenecks_block=args["max_bottlenecks_block"]
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
			# Train the discriminator first (several times)
			for diter in range(args["disc_train_steps"]):
				sess.run(iterator.initializer, feed_dict={filename: train_db_path})
				try:
					while True:
						# Feeding mandatory because it is used to create labels
						temp_loss = sess.run(d_train_op, feed_dict={filename: train_db_path})
						print("Temp d_loss", temp_loss)
				except KeyboardInterrupt:
					# User terminated with Ctrl-C
					should_end = True
					print('Interrupted during epoch ' + str(epoch))
					break
				except tf.errors.OutOfRangeError:
					# Epoch is finished
					print('Discriminator training on epoch', str(epoch), 'finished, iteration', str(diter))
			# Train the generator once
			if not should_end:
				sess.run(iterator.initializer, feed_dict={filename: train_db_path})
				try:
					while True:
						# Feeding mandatory because it is used to create labels
						temp_loss = sess.run(g_train_op, feed_dict={filename: train_db_path})
						print("Temp g_loss", temp_loss)
				except KeyboardInterrupt:
					# User terminated with Ctrl-C
					should_end = True
					print('Interrupted during epoch ' + str(epoch))
					break
				except tf.errors.OutOfRangeError:
					# Update epoch
					new_epoch = epoch + 1
					# Epoch is finished
					print('Generator training on epoch', str(epoch), 'finished')
				
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
				curr_d_acc, curr_g_acc = 0., 0.
				counter = 0
				try:	
					while True:
						loc_d_acc, loc_g_acc = sess.run([d_accuracy, g_accuracy], feed_dict={filename: train_db_path})
						curr_d_acc += loc_d_acc
						curr_g_acc += loc_g_acc
						counter += 1
				except KeyboardInterrupt:
					# User terminated with Ctrl-C
					should_end = True
					print('Evaluation on training set interrupted on epoch ' + str(epoch))
					break
				except tf.errors.OutOfRangeError:
					print('Epoch ' + str(epoch) + ' evaluated on training set')
					# Store summaries
					curr_d_acc /= counter
					curr_g_acc /= counter
					summary_values = [
						tf.Summary.Value(tag="Evaluation/d_accuracy", simple_value=curr_d_acc*100.0),
						tf.Summary.Value(tag="Evaluation/g_accuracy", simple_value=curr_g_acc*100.0)]
					summary = tf.Summary(value=summary_values)
					summary_writer_t.add_summary(summary, global_step=epoch)
					print('Summary for training written on epoch ' + str(epoch))
					
				# Evaluation on validation set
				sess.run(iterator.initializer, feed_dict={filename: valid_db_path})
				curr_d_acc, curr_g_acc = 0., 0.
				counter = 0
				try:
					while True:
						loc_d_acc, loc_g_acc = sess.run([d_accuracy, g_accuracy], feed_dict={filename: valid_db_path})
						curr_d_acc += loc_d_acc
						curr_g_acc += loc_g_acc
						counter += 1
				except KeyboardInterrupt:
					# User terminated with Ctrl-C
					should_end = True
					print('Evaluation on validation set interrupted on epoch ' + str(epoch))
					break
				except tf.errors.OutOfRangeError:
					print('Epoch ' + str(epoch) + ' evaluated on validation set')
					# Store summaries
					curr_d_acc /= counter
					curr_g_acc /= counter
					summary_values = [
						tf.Summary.Value(tag="Evaluation/d_accuracy", simple_value=curr_d_acc*100.0),
						tf.Summary.Value(tag="Evaluation/g_accuracy", simple_value=curr_g_acc*100.0)]
					summary = tf.Summary(value=summary_values)
					summary_writer_v.add_summary(summary, global_step=epoch)
					print('Summary for validation written on epoch ' + str(epoch))
					
			# Update epoch
			epoch = new_epoch
		
	# Time elapsed
	timeElapsed = datetime.now()-startTime
	print('Time elapsed {}'.format(timeElapsed))
			