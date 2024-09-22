#!/bin/bash

# num of test times
n=10

file="/home/code/test_sensing.py"


for i in $(seq 1 $n)
do
  echo "test$i"
  python $file
done

