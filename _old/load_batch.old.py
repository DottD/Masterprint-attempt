#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:59:17 2017

@author: MacD
"""

import os
import matplotlib.pyplot as plt
import random
import numpy as np

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
    if stack:
        # Take n random file names from the input stack
        batch = []
        if n is None:
            local_selection = stack
        else:
            local_selection = random.sample(stack, n)
        for fileName in local_selection:
            img = np.array(plt.imread(fileName))
            img = np.expand_dims(img, axis=2)
            #img3 = np.stack([img, img, img], axis=2)
            batch.append(img) # use img3 to create a 3-channel image
        return np.stack(batch, axis=0)