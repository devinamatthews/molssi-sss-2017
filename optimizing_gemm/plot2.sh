#!/bin/bash

command="set key top left
set terminal x11
set xlabel 'Matrix size (n)'
set ylabel 'GFLOPs'
set xrange [0:1000]
set yrange [0:]
plot 'out_blas' w lines title 'BLAS'"

for step in "$@"; do
    command="$command, 'out_$step' w lines title 'Step $step'"
done

echo "$command" | /usr/local/bin/gnuplot -persist

