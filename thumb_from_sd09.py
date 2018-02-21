"""
Extraction of thumbs from sd09 dataset
"""
import os
import argparse
import shutil
import progressbar
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import scipy.ndimage
import scipy.spatial
import scipy.misc
from peakdetect import peakdet
import math
from multiprocessing import Pool
import psutil


def scan_dir(path, ending):
	"""Recursively scan the folder"""
	file_list = []
	dir_list = []
	for curr_dir, _, local_files in os.walk(path):
		# filter local files
		local_files = [os.path.join(curr_dir, x) for x in local_files if x.endswith(ending)]
		# append to global list
		file_list += local_files
		if local_files:
			dir_list.append(curr_dir)
	return dir_list, file_list
	
def gaussWin2D(shape, sigma=None):
	"""
	Create a 2D Gaussian window
	The shape must have 2 components, namely the vertical and horizontal,
	in this order.
	"""
	# Check input
	if len(shape) == 1:
		shape = (shape, shape)
	elif len(shape) > 2:
		shape = shape[:1]
	shape = [max([1, x]) for x in shape]
	if not sigma:
		sigma = [x/2.0 for x in shape]
	else:
		if len(sigma) == 1:
			sigma = (sigma, sigma)
		elif len(shape) > 2:
			sigma = sigma[:1]
		sigma = [np.finfo(float32).eps for x in sigma if x <= 0]
	# Create vertical and horizontal components
	v = scipy.signal.gaussian(shape[0], sigma[0])
	v = np.reshape(v, (-1, 1)) # column
	h = scipy.signal.gaussian(shape[1], sigma[1])
	h = np.reshape(h, (1, -1)) # row
	return np.dot(v, h)
	
def hannWin2D(shape, sigma=None):
	"""
	Create a 2D Gaussian window
	The shape must have 2 components, namely the vertical and horizontal,
	in this order.
	"""
	# Check input
	if len(shape) == 1:
		shape = (shape, shape)
	elif len(shape) > 2:
		shape = shape[:1]
	shape = [max([1, x]) for x in shape]
	if not sigma:
		sigma = [x/2.0 for x in shape]
	else:
		if len(sigma) == 1:
			sigma = (sigma, sigma)
		elif len(shape) > 2:
			sigma = sigma[:1]
		sigma = [np.finfo(float32).eps for x in sigma if x <= 0]
	# Create vertical and horizontal components
	v = scipy.signal.hann(shape[0], sym=True)
	v = np.reshape(v, (-1, 1)) # column
	h = scipy.signal.hann(shape[1], sym=True)
	h = np.reshape(h, (1, -1)) # row
	return np.dot(v, h)
	
def circleWin(rad):
	rad_span = np.arange(-rad, rad+1)
	x, y = np.meshgrid(rad_span, rad_span)
	kernel = x**2 + y**2 <= rad**2
	return kernel.astype(int)
	
def cropToMaxEnclosedSquare(image):
	s = min(image.shape)
	return image[0:s, 0:s]

def cart2pol(_mat, N):
	"""
	Transform an XY matrix in a rho-theta matrix.
	N is the number of angles used in the process.
	"""
	# Force mat to be a square
	if _mat.shape[0] != _mat.shape[1]:
		mat = cropToMaxEnclosedSquare(_mat)
	mat_size = mat.shape[0]
	mat_center = mat_size/2.0+0.5
	theta_span = np.linspace(0, np.pi, N)
	rad_span = np.arange(0, mat_center)
	theta, rad = np.meshgrid(theta_span, rad_span)
	xx = rad * np.cos(theta) + mat_center
	yy = rad * np.sin(theta) + mat_center
	xy_span = np.arange(0, mat_size)
	interpolator = scipy.interpolate.RectBivariateSpline(xy_span, xy_span, mat)
	return interpolator.ev(xx, yy)
	
def computeRidgeFrequency(image):
	# Compute the FFT of the image using a gaussian window
	kernel = hannWin2D(image.shape)
	img = image * kernel
	img = (img-img.mean()) / img.std()
	img = np.fft.fftshift(np.absolute(np.fft.fft2(img)))
	# Convert the image to polar representation
	img = cart2pol(img, 16)
	# Sum (L2-norm) along the angle axis, to get the energy for each circle
	circle_energy = np.sqrt(np.sum(img**2, axis=1))
	# Find the most energetic circle
	circle_logen = np.log(circle_energy)
	circle_logen = scipy.ndimage.filters.gaussian_filter(circle_logen, sigma=2, mode="nearest")
	peakind, _ = peakdet(circle_logen, circle_logen.std()/2)
	if len(peakind)==0 or peakind[0, 0]==0:
		return 10
	else:
		return img.shape[0]/peakind[0, 0]
		
def segmentation(image, _f, p):
	"""
	Segment input image.
	Args:
		- image: the image to be segmented
		- _f: frequency of the ridges in the given image
		- p: percentile for threshold 
		
		
	"""
	# Take r as half the frequency
	f = math.ceil(_f/2)
	# Generate a circle window as neighborhood
	kernel = circleWin(f)
	kernel = kernel / kernel.sum()
	# Compute the mean in each neighborhood
	m = scipy.signal.convolve2d(image, kernel, mode='same', boundary='wrap')
	# Compute the standard deviation in each neighborhood
	s = np.sqrt(scipy.signal.convolve2d((image-m)**2, kernel, mode='same', boundary='wrap'))
	# Smooth the values
	s = scipy.ndimage.filters.gaussian_filter(s, 8*f, mode='nearest')
	# Compute the threshold for s as the p-percentile
	t = np.percentile(s, p)
	# Segmentation
	foreground = (s >= t).astype(int)
	# Compute the connected components of the foreground and select the largest
	label_im, n_labels = scipy.ndimage.label(foreground)
	all_labels = np.arange(1, n_labels+1)
	label_area = scipy.ndimage.labeled_comprehension(foreground, label_im, all_labels, np.sum, int, 0)
	largest_idx = all_labels[np.argmax(label_area)]
	foreground = (label_im == largest_idx).astype(int)
	return foreground

def random_crop_and_norm(img, img_shape=(128, 128)):
	"""
	Generate partial fingerprints
	"""
	# Crop the image
	y, x = [np.random.randint(0, img.shape[i]-img_shape[i]+1) for i in range(2)]
	return img[y:y+img_shape[0], x:x+img_shape[1]]

if __name__ == '__main__':
	# Parsing arguments
	parser = argparse.ArgumentParser(description="Extraction of thumbs from sd09 dataset")
	parser.add_argument("in", help="Path to the input database directory")
	parser.add_argument("out", help="Path to the output database directory")
	parser.add_argument("-N", "--num-partials", default=15, type=int, help="Number of partial images per finger (train db)")
	parser.add_argument("-V", "--validation-num", default=3, type=int, help="Number of partial images per finger (validation db)")
	parser.add_argument("--keep-class", action="store_true", default=False, help="Whether to keep the last directory before the file")
	parser.add_argument("--no-rnd-crop", action="store_true", default=False, help="Whether to perform only preprocessing and not random cropping")
	parser.add_argument("--img-shape", default=(128, 128, 1), 
		type=lambda strin: tuple(int(val) for val in strin.split('x')),
		help="Expected image shape in the form 'WxHxD', W=width, H=height, D=depth")
	args = vars(parser.parse_args())
	
	# Take each thumb image in the input folder
	in_abs_dir = os.path.abspath(args["in"])
	out_abs_dir = os.path.abspath(args["out"])
	N = args["num_partials"]
	V = args["validation_num"]
	img_shape = args["img_shape"]
	_, thumbs_files = scan_dir(in_abs_dir, "_01.png")
	# Define the processing function
	def processing_fn(file):
		# Decide the output file path
		out_file, out_ext = os.path.basename(file).split('.')
		if not args["no_rnd_crop"]:
			out_dir_train = os.path.join(out_abs_dir, 'train')
			out_dir_val = os.path.join(out_abs_dir, 'val')
			if args["keep_class"]:
				class_name = file.split(os.sep)[-2]
				out_dir_train = os.path.join(out_dir_train, class_name)
				out_dir_val = os.path.join(out_dir_val, class_name)
			if not os.path.exists(out_dir_train):
				os.makedirs(out_dir_train)
			if not os.path.exists(out_dir_val):
				os.makedirs(out_dir_val)
		else:
			if args["keep_class"]:
				class_name = file.split(os.sep)[-2]
				out_dir = os.path.join(out_abs_dir, class_name)
			if not os.path.exists(out_dir):
				os.makedirs(out_dir)
			
		# Read image
		img = plt.imread(file)
		# Compute ridge frequency
		freq = computeRidgeFrequency(img)
		# Segmentation
		mask = segmentation(img, freq, 85)
		# Compute the bounding box (axis oriented) of the points in the mask,
		# then crop the given image on that region.
		row, col = np.nonzero(mask)
		# Enlarge small ROIs
		U, D = min(row), max(row)
		if D-U < img_shape[0]:
			diff = math.ceil( (img_shape[0]-D+U)/2 )
			U = max([U-diff, 0]) # never below 0
			D = U+img_shape[0] # error only if img_shape bigger than original
		L, R = min(col), max(col)
		if R-L < img_shape[1]:
			diff = math.ceil( (img_shape[1]-R+L)/2 )
			L = max([L-diff, 0])
			R = L+img_shape[1]
		# Crop to ROI
		try:
			img = img[U:D, L:R]
		except:
			return

		if not args["no_rnd_crop"]:
			# Check if the dimension of the image is sufficient
			if img.shape[0] >= img_shape[0] and img.shape[1] >= img_shape[1]:
				# Save images for training
				for n in range(N):
					tmp = random_crop_and_norm(img, img_shape)
					scipy.misc.imsave(os.path.join(out_dir_train, out_file+'_'+str(n)+'.'+out_ext), tmp)
				# Save images for validation
				for n in range(V):
					tmp = random_crop_and_norm(img, img_shape)
					scipy.misc.imsave(os.path.join(out_dir_val, out_file+'_'+str(n)+'.'+out_ext), tmp)
			else: print("Relevant part too small for", out_file)
		else:
			scipy.misc.imsave(os.path.join(out_dir, out_file+'.'+out_ext), img)
	
	# Parallel execution
	with Pool(psutil.cpu_count()) as p:
		p.map(processing_fn, thumbs_files)