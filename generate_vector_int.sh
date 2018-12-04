#!/bin/bash

SIZE=$1
FILE=$2

echo $SIZE > $FILE
for i in $(seq 1 $SIZE)
do
  val=$((RANDOM % 256))
  echo $val >> $FILE
done
