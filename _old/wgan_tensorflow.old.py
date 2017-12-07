import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import os
import math

# Define function for recursive scan of the database folder
def scan_dir(path):
    file_list = []
    for curr_dir, local_dirs, local_files in os.walk(path):
        # filter local files
        local_files = [os.path.join(path,x) for x in local_files if x.endswith('.png')]
        # append to global list
        file_list += local_files
        # recursively scan each subdirectory
        for inner_dir in local_dirs:
            file_list += scan_dir(inner_dir)
    return file_list

# Function to load another batch of images from a stack (list)
def load_random_batch(stack, n):
    # Check if the stack is empty
    if not stack:
        raise StandardError('The input list of files cannot be empty')
    # Take n random file names from the input stack
    batch = []
    for fileName in random.sample(stack, n):
        img = plt.imread(fileName)
        flat_img = [element for row in img for element in row]
        batch.append(flat_img)
    return batch

# Visualization purposes
def plot(gen, n_images, im_dim, out_path, idx):
    # Compute how many rows and cols of images should visualize
    rows = math.floor(math.sqrt(n_images))
    cols = math.ceil(math.sqrt(n_images))
    # Generate images
    z_mb = sample_z(rows*cols, z_dim)
    samples = sess.run(gen, feed_dict={z: z_mb})
    # Create figure and plot sample images
    fig = plt.figure(figsize=(rows, cols))
    gs = gridspec.GridSpec(rows, cols)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(im_dim[0], im_dim[1]), cmap='Greys_r')
    
    # Save image to file
    file_name = os.path.join(out_path,'{}.png').format(str(idx).zfill(3))
    plt.savefig(file_name, bbox_inches='tight')
    plt.close(fig)
    return idx+1

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])
#--------------------------

# Function that initializes a vector with xavier distribution
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# Parameters
n_images = 16 # number of images to visualize at each session
mb_size = 32 # number of images per training session (mini batch size)
im_dim = [128,128]
X_dim = im_dim[0]*im_dim[1] # discriminator input dimension
z_dim = 100 # generator input dimension
h_dim = 128 # kind of "amount of internal parameters"
db_path = '/Users/MacD/Databases/mod_sd09' # Path to the database folder
out_folder ='training_steps/' # Name of the folder with training outputs


# Recursively scan the db folder to find .png files
file_list = scan_dir(db_path)
out_path = os.path.join(os.path.dirname(db_path), out_folder)
if not os.path.exists(out_path):
    os.makedirs(out_path)

# Placeholders for batches of images (X) and generator inputs (z)
X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

# Discriminator internal variables
D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))
theta_D = [D_W1, D_W2, D_b1, D_b2]

# Generator internal variables
G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))
theta_G = [G_W1, G_W2, G_b1, G_b2]

#Define generator based on internal variables theta_G
def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob

#Define generator based on internal variables theta_D
def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    out = tf.matmul(D_h1, D_W2) + D_b2
    return out

# Define the scheme of the network
G_sample = generator(z)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(-D_loss, var_list=theta_D))
G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(G_loss, var_list=theta_G))

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

# Initialize session and variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Initialize counter for visualization
i = 0

for it in range(1000000):
    # Train the discriminator first, 5 times
    for _ in range(5):
        # Generate or load feeding data
        X_mb = load_random_batch(file_list, mb_size)
        z_mb = sample_z(mb_size, z_dim)
        # Train the discriminator
        _, D_loss_curr, _ = sess.run(
            [D_solver, D_loss, clip_D],
            feed_dict={X: X_mb, z: z_mb}
        )

    # Train the generator
    z_mb = sample_z(mb_size, z_dim)
    _, G_loss_curr = sess.run(
        [G_solver, G_loss],
        feed_dict={z: z_mb}
    )

    if it % 10 == 0:
        print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss_curr, G_loss_curr))

        if it % 50 == 0:
            i = plot(G_sample, n_images, im_dim, out_path, i)
