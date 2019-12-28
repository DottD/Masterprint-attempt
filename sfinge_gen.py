#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import scipy.ndimage
import scipy.signal
import skimage.filters
import matplotlib.pyplot as plt
from math import tan, atan, cos, sin, sqrt, log, exp, ceil
from cmath import exp
from collections import deque
from datetime import datetime 

		
def vis_orient(theta):
	rows, cols = theta.shape
	x, y = np.meshgrid(np.arange(cols), np.arange(rows))
	S = 6
	x = x[::S,::S]
	y = y[::S,::S]
	theta = theta[::S,::S]
	plt.quiver(x, y, np.cos(theta), np.sin(theta),
		color='b', units='xy', pivot='mid', angles='uv',
		# with 'uv' inverting axis does not invert arrows
		headaxislength=0, headlength=0)
	
def rand_norm_int(low, high, size=None):
	r = np.random.normal(loc=(high+low)/2., scale=(high-low)/6., size=size)
	r = np.minimum(r, high)
	r = np.maximum(r, low)
	return r.astype(int)
	
class FPGenerator:
	"""
	Fingerprint generator class:
		- init: initializes the model for performance improvements
		- call: generate fingerprints from given parameters
		- random: returns some "meaningful" parameters
	"""
	L = 8 # fixed number of orientation correction parameters per singularity
	T = 9 # spatial period
	
	def __init__(self, **args):
		# Read parameters
		self.rows = args["rows"]
		self.cols = args["cols"]
		if self.rows <= 0 or self.cols <= 0:
			raise ValueError("Shape must contain only positive values")
		# Initialization - orientation
		x, y = np.meshgrid(np.arange(self.cols), np.arange(self.rows))
		self.z = x + y * 1j
		self.ai = np.linspace(-np.pi, np.pi, num=self.L+1, endpoint=True)
		# Initialization - ridge generation
		self.images = [np.zeros((self.rows, self.cols)), np.zeros((self.rows, self.cols))]
		# Compute parameters
		f = 1/self.T # spatial frequency
		sigma = 3/(2*f) * sqrt(1./(6*log(10))) # standard deviation of gabor filter's gaussian envelope
		self.N = 36 # number of angular samples
		R = self.T # half side of the gabor filter
		x, y = np.meshgrid(np.arange(-R, R+1), np.arange(-R, R+1))
		self.s = self.T # breadth of normalization kernel
		self.gen_iterations = 1000 # total number of filter applications
		# Create the filter banks
		self.angle_samples = np.linspace(-np.pi/2, np.pi/2, num=self.N, endpoint=False)
		self.gabor = []
		for t in self.angle_samples:
			# Define filter
			tt = t+np.pi/2 # sinusoid along the orientation, not orthogonal
			self.gabor.append( np.exp( -(x**2+y**2)/(2*sigma**2) ) * np.cos( 2*np.pi*f * (x*cos(tt)+y*sin(tt)) ) )
	
	# Define the gk function (depends only on the alpha_i parameters)
	def __gk(self, alpha, noise):
		# Compute left and right extrema as pairs like
		# (new value, old value)
		l = max([ [a[1]+noise[a[0]%self.L], a[1]] for a in enumerate(self.ai) if alpha >= a[1]], key=lambda t:t[1])
		r = min([ [a[1]+noise[a[0]%self.L], a[1]] for a in enumerate(self.ai) if alpha <= a[1]], key=lambda t:t[1])
		if l == r: return l[0]
		else: return l[0] + (alpha-l[1]) / (r[1]-l[1]) * (r[0]-l[0])

	
	def orientation_image(self, **args):
		"""
		Generation of the orientation image
		Args:
			- loops, [[ls_1 x, ls_1 y, ls_1 g_1, ..., ls_1 g_8], [ls_2 x, ls_2 y, ls_2 g_1, ..., ls_2 g_8]]
			- deltas, same format as loops
		"""
		loops = args["loops"]
		deltas = args["deltas"]
		# Check input
		if any([len(row)!=(self.L+2) for row in loops]) or any([len(row)!=(self.L+2) for row in deltas]):
			raise ValueError("Each singularity must be defined by 2 coordinates and "+str(self.L)+" correction parameters")
		# Generate orientation
		theta = np.zeros(self.z.shape)
		for loop in loops:
			ls = loop[0] + loop[1] * 1j
			gk = np.frompyfunc(lambda alpha:self.__gk(alpha, loop[2:]), 1, 1)
			theta += gk(np.angle(self.z-ls)).astype(float)
		for delta in deltas:
			ds = delta[0] + delta[1] * 1j
			gk = np.frompyfunc(lambda alpha:self.__gk(alpha, delta[2:]), 1, 1)
			theta -= gk(np.angle(self.z-ds)).astype(float)
		while theta.min() < 0: theta += 2*np.pi
		theta += np.pi # compensate the final minus operation
		theta %= 2*np.pi # angle circularity
		theta -= np.pi # back in the range [-pi, pi)
		return theta/2
		
	def ridge_pattern(self, **args):
		"""
		Create ridge pattern from orientation map and initial spots
		Args:
			- theta: orientation map as returned from orientation_image
			- spots: [[x1, y1], [x2, y2], ...] list of spots coordinates
		"""
		theta = args["theta"]
		spots = args["spots"]
		if "pause" in args:
			pause = args["pause"]
		else:
			pause = None
		# Add some spots to the image
		spots = np.array(spots).astype(int)
		__old = self.images[0]
		__new = self.images[1]
		__old.fill(0) # reset image 1
		__new.fill(0) # reset image 2
		__old[theta.shape[0]-1-spots[:,1], spots[:,0]] = 1
		# Convert theta and generate masks
		Theta = np.exp(2j*theta)
		ang_dist = lambda t: 0.5 * np.arccos( 1 - np.absolute(exp(2j*t)-Theta)**2 / 2 )
		dt = np.pi/self.N # step between two consecutive angular samples
		mask = []
		for t in self.angle_samples:
			# Define a mask
			mask.append( skimage.filters.gaussian((ang_dist(t) <= dt/2).astype(float), sigma=2) )
		# Apply the filters
		min_allowed = theta.size * 0.01
		differences = min_allowed+1
		while differences > min_allowed:
			# Apply filter bank
			for n in range(self.N):
				__new += scipy.signal.convolve(__old, self.gabor[n], mode='same') * mask[n]
			# Clipping
			__new[__new > 1] = 1
			__new[__new < -1] = -1
			# Count differences
			differences = np.count_nonzero(np.abs(__new-__old)>0)
			# Visualizationm
			if pause:
				plt.ion()
				plt.cla()
				plt.imshow(__new, cmap='Greys', origin='lower')
				vis_orient(theta)
				plt.title('Differences '+str(differences))
				plt.pause(pause)
			# Update image and reset new_image
			__new, __old = __old, __new
			
		return __new
		
	def random(self, verbose=False, nc=None, nd=None, n_spots=None):
		# Choose the orientation parameters
		S = 4
		def gen_noise():
			var = np.pi/2
			noise = np.zeros((self.L))
			noise[::S] = var*(2*np.random.rand(self.L//S)-1)
			return scipy.ndimage.filters.gaussian_filter1d(noise, sigma=1)
		C = (self.rows//2, self.cols//2)
		if not nc: nc = rand_norm_int(low=1, high=2)
		if not nd: nd = rand_norm_int(low=1, high=2)
		if verbose: print(nc, "loops:")
		loops = []
		for _ in range(nc): # define loops
			loop = [C[1] + rand_norm_int(low=-2*C[1], high=2*C[1])]
			loop.append( C[0] + rand_norm_int(low=-2*C[0], high=2*C[0]) )
			loop.extend(gen_noise())
			loops.append(loop)
			if verbose: print("  ", loop)
		if verbose: print(nd, "deltas:")
		deltas = []
		for _ in range(nd): # define deltas
			delta = [C[1] + rand_norm_int(low=-2*C[1], high=2*C[1])]
			delta.append( C[0] + rand_norm_int(low=-2*C[0], high=2*C[0]) )
			delta.extend(gen_noise())
			deltas.append(delta)
			if verbose: print("  ", delta)
		# Choose the generation parameters
		spots = []
		if not n_spots: n_spots = rand_norm_int(low=1, high=20)
		if verbose: print(n_spots, "spots:")
		for _ in range(n_spots):
			spots.append([np.random.randint(low=0, high=self.cols), np.random.randint(low=0, high=self.rows)])
			if verbose: print("  ", spots[-1])
			
		return loops, deltas, spots
			
	def __call__(self, *args):
		loops = args[0]
		deltas = args[1]
		spots = args[2]
		if len(args) == 4:
			pause = args[3]
		else:
			pause = None
		theta = self.orientation_image(loops=loops, deltas=deltas)
		return self.ridge_pattern(theta=theta, spots=spots, pause=pause), theta

if __name__ == '__main__':
	# Create fingerprints
	rows = 128
	cols = 128
	gen = FPGenerator(rows=rows, cols=cols)
	# Generate fingerprint
#	loops, deltas, spots = gen.random()
	loops = [[77, 115, 0.7232385865949174, 0.3300394842483926, 0.01727738832468644, -0.21292984344285856, -0.3594067792455023, -0.21808284493201732, -0.048660860115306216, -0.004114944848531915]]
	deltas = [[-19, 190, -0.5978972322369098, -0.2728772302189597, -0.014721155985321189, 0.17406430997080802, 0.2938821307740076, 0.17832425930122192, 0.039789520536231776, 0.0033647510990998737]]
	spots = [[64, 100]]
	image, theta = gen(loops, deltas, spots)
	# Visualization
	plt.ioff()
	plt.imshow(image, cmap='Greys', origin='lower')
	vis_orient(theta)
	plt.plot([loop[0] for loop in loops], [loop[1] for loop in loops], linestyle=' ', marker='o', color='r', markersize=10)
	plt.plot([delta[0] for delta in deltas], [delta[1] for delta in deltas], linestyle=' ', marker='^', color='r', markersize=10)
	plt.plot([spot[0] for spot in spots], [spot[1] for spot in spots], linestyle=' ', marker='.', color='c')
	plt.axis('image')
	plt.xlim(0, cols)
	plt.ylim(0, rows)
	plt.show()