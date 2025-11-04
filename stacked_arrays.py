#!/usr/bin/env python3
#import required modules
import numpy as np
import matplotlib.pyplot as plt

#create two differently sized arrays
array1 = np.random.rand(24)
array2 = np.random.rand(23)

#define x-axis positions
x1 = np.arange(len(array1))
x2 = np.arange(len(array2)) + 0.5

#define y-axis levels
y1 = np.zeros_like(array1)
y2 = np.ones_like(array2)

#plot the arrays
plt.figure(figsize = (8, 2))
plt.scatter(x1, y1, color = 'blue', label = 'Array 1', s = 100)
plt.scatter(x2, y2, color = 'red', label = 'Array 2', s = 100)

#format the plot
plt.xticks([])
plt.yticks([])

#%%try visualization using pcolormesh
grid = np.full((2, 24), np.nan)

#assign values to grid
grid[1, :] = array1
