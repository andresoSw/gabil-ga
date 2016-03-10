#!/bin/bash
# script for executing i times the gabil learning algorithm
# BUG: number of generations -g must be 100 or higher, or else not all the runs will be
# saved in different folders

for i in `seq 1 10`;
	do
		python main.py -c 0.06 -m 0.01 -d datasets/crx.data -p 8 -g 100 --rfolder my_test
	done
