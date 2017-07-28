import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import glob

# Parse some args
parser = argparse.ArgumentParser(description='A simple script to make a line plot.')
parser.add_argument('--test', action='store_true', help='Test the plotting tool with fake data.')
parser.add_argument(
    '--file',
    type=str,
    default=None,
    help=
    "Filename to parse the x and y coordinates from. Filenames can be seperated by commas if multiple"
    "files are desired. If None, defaults to all '*.gemm' files in the current folder."
)
parser.add_argument(
    '--ylabel', type=str, default="GFLOPS", help='Filename to parse the x and y coordinates from.')
parser.add_argument(
    '--xlabel', type=str, default="n", help='Filename to parse the x and y coordinates from.')
parser.add_argument(
    '--output',
    type=str,
    default="plot",
    help='Filename to parse the x and y coordinates from.')
args = parser.parse_args()

args, unknown = parser.parse_known_args()

# We have unparsed args
if len(unknown):
    raise ParseError("The following keywords are not recognized: %s" % ", ".join(unknown))

# Lets grab a plain dict
args = args.__dict__

# Obtain out data
if args["test"]:
    data = [np.array([[2, 4, 6], [4, 16, 36]]).T, np.array([[2.5, 3.5, 6], [5, 11, 22]]).T]
    args["names"] = ["test1", "test2"]
else:

    # Figure out the filenames
    if args["file"] is None:
        files = glob.glob("*.gemm")
    else:
        files = args["file"].split(",")

    args["names"] = [x.replace(".gemm", "") for x in files]

    # Parse in the data
    data = []
    for fname in files:
        if not os.path.isfile(fname):
            raise NameError("File %s does not exist" % fname)
        tmp = np.loadtxt(fname, delimiter=" ")
        if tmp.ndim != 2:
            raise AttributeError("Input must have exactly two dimensions.")

        if tmp.shape[1] != 2:
            raise AttributeError("Input must have exactly two columns.")

        data.append(tmp)

# Make our plot
handles = []
for fname, d in zip(args["names"], data):
    tmp = plt.plot(*d.T, label=fname)
    handles.append(tmp[0])
plt.ylabel(args["ylabel"])
plt.xlabel(args["xlabel"])
plt.legend(handles=handles)

# Write it out
plt.savefig(args["output"] + '.png')
