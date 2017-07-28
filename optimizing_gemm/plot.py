#!/usr/bin/env python

import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

# Parse some args
parser = argparse.ArgumentParser(description='A simple script to make a line plot.')
parser.add_argument('nums', metavar='N', type=int, nargs='+',
                    help='step to plot')
args = parser.parse_args()

plots = []
for num in args.nums + ["blas"]:
    filename = "out_"+str(num)
    if not os.path.isfile(filename):
        raise NameError("File %s does not exist" % filename)

    data = np.loadtxt(filename, delimiter=" ")
    if data.ndim != 2:
        raise AttributeError("Input must have exactly two dimensions.")

    if data.shape[1] != 2:
        raise AttributeError("Input must have exactly two columns.")

    plots.append(data)

# Make our plot
handles = []
for num, data in zip(args.nums, plots):
    tmp = plt.plot(*data.T, label="Step "+str(num))
    handles.append(tmp[0])
tmp = plt.plot(*plots[-1].T, label="BLAS")
handles.append(tmp[0])
plt.ylabel("GFLOPs")
plt.xlabel("Matrix size (n)")
plt.legend(handles=handles)
plt.ylim(ymin=0)

# Figure out the output name
outname = "plot_" + "".join(map(str, args.nums))

# Write it out
plt.savefig(outname + '.png')
