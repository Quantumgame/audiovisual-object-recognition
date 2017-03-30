#!/bin/bash

while read object
do 
	for i in {0..49}
	do
		if [ ! -f ./bag/$object-$i-rgb.png ]; then
			echo $object-$i-rgb.png 'not found!!'
		fi
		if [ ! -f ./bag/$object-$i-depth.png ]; then
			echo $object-$i-depth.png 'not found!!'
		fi
	done
done < $1