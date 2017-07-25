#!/bin/bash

export GNUPLOT_PS_DIR="$HOME/miniconda/envs/sss/share/gnuplot/4.6/PostScript"

file1="plot_`echo $* | tr -d ' '`.ps"
file2="plot_`echo $* | tr -d ' '`.pdf"
command="set key top left
set terminal postscript
set output '$file1'
set xlabel 'Matrix size (n)'
set ylabel 'GFLOPs'
set xrange [0:1000]
set yrange [0:]
plot 'out_blas' w lines title 'BLAS'"

for step in "$@"; do
    command="$command, 'out_$step' w lines title 'Step $step'"
done

echo "$command" | gnuplot
ps2pdf $file1
xdg-open $file2 || open $file2

