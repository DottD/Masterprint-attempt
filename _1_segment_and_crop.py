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
from scipy.ndimage.filters import gaussian_gradient_magnitude
from peakdetect import peakdet
import math
from multiprocessing import Pool
import psutil
import h5py
from progressbar import widgets


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
	# Compute image edges
	s = scipy.ndimage.gaussian_gradient_magnitude(image, sigma=_f//6, mode='nearest')
	# Normalize image globally
	s = (s-s.min())/(s.max()-s.min())
	# Saturate values far from the threshold
	rad = min([p, 1-p])/2
	saturate_fn = lambda X: (np.tanh(2*(X-p)/rad)+1)/2
	# Saturate the image and smooth the image
	s = scipy.ndimage.filters.gaussian_filter(saturate_fn(s), _f//2, mode='nearest')
	# Segmentation
	foreground = (s >= p).astype(int)
#	# Take r as half the frequency
#	f = math.ceil(_f/2)
#	# Generate a circle window as neighborhood
#	kernel = circleWin(f)
#	kernel = kernel / kernel.sum()
#	# Compute the mean in each neighborhood
#	m = scipy.signal.convolve2d(image, kernel, mode='same', boundary='symm')
#	# Compute the standard deviation in each neighborhood
#	s = np.sqrt(scipy.signal.convolve2d((image-m)**2, kernel, mode='same', boundary='symm'))
#	# Smooth the values
#	s = scipy.ndimage.filters.gaussian_filter(s, 8*f, mode='nearest')
#	# Compute the threshold for s as the p-percentile
#	t = np.percentile(s, p)
#	# Segmentation
#	foreground = (s >= t).astype(int)
	# Compute the connected components of the foreground and select the largest
	label_im, n_labels = scipy.ndimage.label(foreground)
	all_labels = np.arange(1, n_labels+1)
	label_area = scipy.ndimage.labeled_comprehension(foreground, label_im, all_labels, np.sum, int, 0)
	largest_idx = all_labels[np.argmax(label_area)]
	foreground = (label_im == largest_idx).astype(int)
	# Hole fill
	foreground = scipy.ndimage.morphology.binary_fill_holes(foreground)
	return foreground
	
def find_roi_pos(mask, roi_shape=(128,128), step=None):
	# Eventually reduce mask to gridded points
	if step:
		_mask = np.zeros_like(mask)
		_mask[::step, ::step] = np.logical_or(_mask[::step, ::step], mask[::step, ::step])
	else:
		_mask = mask
	# Discard a roi if it has some background points
	rows = []
	cols = []
	for T, L in zip(*np.nonzero(_mask)):
		B = T+roi_shape[0]
		R = L+roi_shape[1]
		if B < _mask.shape[0] and R < _mask.shape[1]:
			roi = mask[T:B, L:R]
			if roi.all(): 
				rows.append(T)
				cols.append(L)
	return rows, cols

def random_crop_and_norm(img, img_shape=(128, 128)):
	"""
	Generate partial fingerprints
	"""
	# Crop the image
	y, x = [np.random.randint(0, img.shape[i]-img_shape[i]+1) for i in range(2)]
	return img[y:y+img_shape[0], x:x+img_shape[1]]

if __name__ == '__main__':
	# Parsing arguments
	parser = argparse.ArgumentParser(description="Extraction of thumbs from sd09 dataset."
		"Please note that images belonging to the same subject must be in the same"
		"subdirectory; the subdir's name will be thought of as the subject name.")
	parser.add_argument("in", help="Path to the input database directory")
	parser.add_argument("out", help="Path to the output database file")
	parser.add_argument("--img-shape", default=(128, 128, 1), 
		type=lambda strin: tuple(int(val) for val in strin.split('x')),
		help="Expected image shape in the form 'WxHxD', W=width, H=height, D=depth")
	args = vars(parser.parse_args())
	
	# Progress bar widgets
	pbwidg = [widgets.SimpleProgress(format=u'%(value_s)s/%(max_value_s)s'), ' ',
			widgets.Bar(marker=u'\u2588'), ' ', widgets.Timer(), ' ', widgets.AdaptiveETA()]
	
	# Name the input arguments
	in_abs_dir = os.path.abspath(args["in"])
	out_abs_dir = os.path.abspath(args["out"])
	img_shape = args["img_shape"]
	# Scan the directory to find the proper files
#	_, thumbs_files = scan_dir(in_abs_dir, "_01.jpg")
	_, thumbs_files = scan_dir(in_abs_dir, ".jpg")
	num_classes = len(thumbs_files)
	# Take the directory name as database name
	db_name = os.path.basename(in_abs_dir)
	# Eventually delete a previous group with the same name
	with h5py.File(out_abs_dir) as f:
		if db_name in f.keys():
			del f[db_name]
	out_abs_dir_tmp = os.path.join(os.path.dirname(out_abs_dir), "tmp.h5")
	os.rename(out_abs_dir, out_abs_dir_tmp)
	os.system("h5repack "+out_abs_dir_tmp+" "+out_abs_dir)
	os.remove(out_abs_dir_tmp)
	# Open storage file (read-write mode)
	with h5py.File(out_abs_dir) as f:
		# Create a group to hold the current database
		grp = f.require_group(db_name)
		# Add the roi size attribute (width, height)
		grp.attrs['roi_size'] = (img_shape[1], img_shape[0])
		# Loop over the files found
		for class_idx, file in progressbar.ProgressBar(max_value=num_classes, widgets=pbwidg)(enumerate(thumbs_files)):
			# Read image
			img = plt.imread(file)
			# Compute ridge frequency
			freq = computeRidgeFrequency(img)
			# Segmentation
			mask = segmentation(img, freq, 0.3)
			# Crop image and mask to mask's bounding box
			rows, cols = np.nonzero(mask)
			mask = mask[min(rows):max(rows), min(cols):max(cols)]
			img = img[min(rows):max(rows), min(cols):max(cols)]
			# Find all the possible top-left corner for partial fingerprints
			tops, lefts = find_roi_pos(mask, roi_shape=img_shape[:-1], step=12)
			# Save image as array of uint8
			imgdt = np.dtype('uint8')
			if not 'images' in grp.keys():
				db = grp.create_dataset(name='images', 
					shape=(num_classes,), dtype=h5py.special_dtype(vlen=imgdt),
					compression="gzip", compression_opts=9)
			else: db = grp['images']
			img = img.astype(np.float64)
			img -= img.min()
			img *= 255/img.max()
			img = img.astype('uint8')
			db[class_idx] = img.flatten()
			# Save image first dimension
			if not 'rows' in grp.keys():
				db = grp.create_dataset(name='rows', 
					shape=(num_classes,), dtype=np.int32,
					compression="gzip", compression_opts=9)
			else: db = grp['rows']
			db[class_idx] = img.shape[0]
			# Save classes' names
			if not 'classes' in grp.keys():
				db = grp.create_dataset(name='classes', 
					shape=(num_classes,), dtype=h5py.special_dtype(vlen=str),
					compression="gzip", compression_opts=9)
			else: db = grp['classes']
			class_name = file.split(os.sep)[-2]
			db[class_idx] = class_name
			# Save possible ROIs positions
			tupdt = np.dtype(np.int32)
			if not 'topleft' in grp.keys():
				db = grp.create_dataset(name='topleft', 
					shape=(num_classes,),  dtype=h5py.special_dtype(vlen=tupdt),
					compression="gzip", compression_opts=9)
			else: db = grp['topleft']
			db[class_idx] = np.dstack((tops, lefts)).flatten().astype(tupdt)
		
#	# Parallel execution
#	with Pool(psutil.cpu_count()) as p:
#		p.map(processing_fn, thumbs_files)
