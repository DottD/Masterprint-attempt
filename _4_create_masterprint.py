#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import argparse
from time import time
import numpy as np
import scipy as sp
import scipy.misc as spm
import cma
from cma.constraints_handler import BoundTransform
from cma import fitness_transformations as ft
from keras.applications import ResNet50
from keras.models import Model, load_model
from sfinge_gen import FPGenerator
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Train a classifier for fingerprints")
parser.add_argument("model", help="Full path to the saved classifier model")
parser.add_argument("out", help="Name of the output folder")
parser.add_argument("--nc", default=1, type=int, help="Number of loops")
parser.add_argument("--nd", default=1, type=int, help="Number of deltas")
parser.add_argument("--ns", default=10, type=int, help="Number of spots")
parser.add_argument("-V", "--verbose", default=False, action="store_true", help="Verbosity")
args = vars(parser.parse_args())

nc = args["nc"]
nd = args["nd"]
ns = args["ns"]
rows = 128
cols = 128
C = (cols//2, rows//2)
generator = FPGenerator(rows=rows, cols=cols)
L = generator.L
verbose = args["verbose"]

# Create the neural network, loading weights from input file and imagenet database
resnet = ResNet50(include_top=False,
		weights='imagenet',
		input_shape=(200,200,3),
		pooling='avg')
model = load_model(args["model"], custom_objects={'binary_sparse_softmax_cross_entropy': 'binary_crossentropy'})
model = Model(inputs=resnet.inputs, outputs=model(resnet.outputs))
model.compile(loss='mse', optimizer='rmsprop')

# Define some useful function
def verification_system(img):
	img = spm.imresize(img, (200, 200))
	img = np.expand_dims(img, axis=2)
	img = np.tile(img, (1,1,3))
	img = np.expand_dims(img, axis=0)
	return model.predict(img)
	
def activated(prediction):
	return np.asscalar(np.mean(prediction, axis=-1))
	
def flatten(x):
	out = []
	for struct in x:
		for X in struct:
			out.append( 2 * (X[0]-C[0])/cols )
			out.append( 2 * (X[1]-C[1])/rows ) # [-1,1] if inside the image
			if len(X) > 2: 
				out.extend( [y/np.pi*2 for y in X[2:]] ) # [-pi/2,pi/2] --> [-1,1]
	return out

def recover(x):
	X = x.tolist()
	loops = []
	for k in range(nc):
		loops.append( X[k*(L+2):(k+1)*(L+2)] )
		loops[-1][0] = int(loops[-1][0]*cols/2 + C[0])
		loops[-1][1] = int(loops[-1][1]*rows/2 + C[1])
		loops[-1][2:] = [y*np.pi/2 for y in loops[-1][2:]]
	deltas = []
	for k in range(nc, nc+nd):
		deltas.append( X[k*(L+2):(k+1)*(L+2)] )
		deltas[-1][0] = int(deltas[-1][0]*cols/2 + C[0])
		deltas[-1][1] = int(deltas[-1][1]*rows/2 + C[1])
		deltas[-1][2:] = [y*np.pi/2 for y in deltas[-1][2:]]
	spots = []
	for k in range((nc+nd)*(L+2), len(x), 2):
		spots.append( X[k:k+2] )
		spots[-1][0] = int(spots[-1][0]*cols/2 + C[0])
		spots[-1][1] = int(spots[-1][1]*rows/2 + C[1])
	return loops, deltas, spots
	
if verbose: plt.figure(figsize=(11, 11*9/16))
def bloody_fn(x):
	loops, deltas, spots = recover(x)
	img = generator(loops, deltas, spots)[0]
	logits = verification_system(img).flatten()
	out = -activated(logits)
	if verbose:
		plt.ion()
		plt.subplot(121)
		plt.cla()
		plt.imshow(img, cmap='Greys', origin='lower')
		plt.plot([loop[0] for loop in loops], [loop[1] for loop in loops], linestyle=' ', marker='o', color='r', markersize=10)
		plt.plot([delta[0] for delta in deltas], [delta[1] for delta in deltas], linestyle=' ', marker='^', color='r', markersize=10)
		plt.plot([spot[0] for spot in spots], [spot[1] for spot in spots], linestyle=' ', marker='.', color='c')
		plt.axis('image')
		plt.xlim(0, cols)
		plt.ylim(0, rows)
		plt.subplot(122)
		plt.cla()
		plt.stem(logits)
		plt.title('Activated = '+str(out))
		plt.pause(1e-3)
	return out

# Create bounds
dim = (nc+nd)*(L+2)+2*ns
bounds = [[-1 for _ in range(dim)], [1 for _ in range(dim)]]
for k in range(nc):
	idx = k*(L+2)
	bounds[0][idx:idx+2] = [None, None]
	bounds[1][idx:idx+2] = [None, None]
for k in range(nc, nc+nd):
	idx = k*(L+2)
	bounds[0][idx:idx+2] = [None, None]
	bounds[1][idx:idx+2] = [None, None]
idx = (nc+nd)*(L+2)
bounds[1][idx::2] = [2*(cols-1-C[0])/cols for _ in range((dim-idx)//2)]
bounds[1][idx+1::2] = [2*(rows-1-C[1])/rows for _ in range((dim-idx)//2)]

# Compute optimal value
x0 = [0 for _ in range(dim)]
result = cma.fmin(bloody_fn, x0, 0.5, {
	'BoundaryHandler': cma.BoundPenalty,
	'bounds': bounds,
	'verb_time': True,
	'verb_disp': 1,
	'verb_log': 0,
	'verbose': 3
	})
for r in result: print(r)