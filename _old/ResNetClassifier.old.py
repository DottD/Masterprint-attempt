import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1
slim = tensorflow.contrib.slim


def resnet_model_fn(features, labels, mode, params):
	"""
	# features: A dict containing the features passed to the model via input_fn
	# labels: A Tensor containing the labels passed to the model via input_fn. Will be empty for predict() calls, as these are the values the model will infer
	# mode: One of the following tf.estimator.ModeKeys string values indicating the context in which the model_fn was invoked:
	# - tf.estimator.ModeKeys.TRAIN
	# - tf.estimator.ModeKeys.EVAL
	# - tf.estimator.ModeKeys.PREDICT
	# params: argument containing a dict of hyperparameters used for training
	# - num_classes: the total number of people enrolled in the dataset
	# - learning_rate: the learning rate for the Adam optimizer
	"""
	with tf.variable_scope('resnet', reuse=False):
		global_step = tf.get_variable(initializer=tf.constant(0),
									trainable=False,
									dtype=tf.int32,
									name='global_step')
		
	# Set up the ResNet architecture
	predictions, _ = resnet_v1.resnet_v1_101(features,
									num_classes=params["num_classes"],
									is_training=True,
									global_pool=True,
									output_stride=None,
									reuse=tf.AUTO_REUSE,
									scope='resnet')
	predictions = tf.reshape(predictions, [tf.shape(features)[0], -1])
	
	# Provide an estimator spec for `ModeKeys.PREDICT`.
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(
				mode=mode,
				predictions={"people": predictions})
	
	# Define the loss function
	
	loss = slim.losses.softmax_cross_entropy(predictions, labels)
	tf.summary.scalar("loss", loss)
	
	# Training operation
	optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"],
									name='Adam')
	train_op = slim.learning.create_train_op(loss, optimizer)
	
	slim.learning.train(
		train_op,
		log_dir,
		number_of_steps=1000,
		save_summaries_secs=300,
		save_interval_secs=600)
	
	return tf.estimator.EstimatorSpec(mode=mode, 
									loss=loss,
									train_op=train_op)
									
def ResNetEstimator(learning_rate=0.00005,
					num_classes=1000,
					model_dir=None,
					config=None):
	"""
	Initialize a ResNet Estimator with given learning rate and output number of classes.
	Args: 
		params: argument containing a dict of hyperparameters used for training
		 - num_classes: the total number of people enrolled in the dataset
		 - learning_rate: the learning rate for the Adam optimizer
	"""
	return tf.estimator.Estimator(model_fn=resnet_model_fn,
								params={"learning_rate":learning_rate, "num_classes":num_classes},
								model_dir=model_dir,
								config=config)