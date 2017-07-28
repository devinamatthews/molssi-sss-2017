import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

# Parse some args
parser = argparse.ArgumentParser(description='A simple script to make a line plot.')
parser.add_argument('--test', action='store_true', help='Test the plotting tool with fake data.')
parser.add_argument(
    '--file', type=str, default=None, help='Filename to parse the x and y coordinates from.')
parser.add_argument(
    '--ylabel', type=str, default="GFLOPS", help='Filename to parse the x and y coordinates from.')
parser.add_argument(
    '--xlabel', type=str, default="n", help='Filename to parse the x and y coordinates from.')
parser.add_argument(
    '--output', type=str, default=None, help='Filename to parse the x and y coordinates from.')
args = parser.parse_args()

args, unknown = parser.parse_known_args()

# We have unparsed args
if len(unknown):
    raise ParseError("The following keywords are not recognized: %s" % ", ".join(unknown))

# Lets grab a plain dict
args = args.__dict__

# Obtain out data
if args["test"]:
    data = np.array([[2, 4, 6], [4, 16, 36]]).T
    args["file"] = "test"
else:
    if not os.path.isfile(args["file"]):
        raise NameError("File %s does not exist" % args["file"])

    data = np.loadtxt(args["file"], delimiter=" ")
    if data.ndim != 2:
        raise AttributeError("Input must have exactly two dimensions.")

    if data.shape[1] != 2:
        raise AttributeError("Input must have exactly two columns.")

# Make our plot
plt.plot(*data.T)
plt.ylabel(args["ylabel"])
plt.xlabel(args["xlabel"])

# Figure out the output name
outname = args["output"]
if outname is None:
    outname = args["file"].split(".")[0]

# Write it out
plt.savefig(outname + '.png')
