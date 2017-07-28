#!/bin/bash

file2="plot_`echo $* | tr -d ' '`.png"
./plot.py $*
xdg-open $file2 || open $file2

