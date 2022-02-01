#!/bin/bash

max_x=$1
echo "Disable $max_x"
i=1
for x in /sys/devices/system/cpu/cpu[1-9]*/online; do
  echo $i
  if [[ "$i" -ge "$max_x" ]];
  then
    break
  fi
  echo 0 >"$x"
  ((i=i+1))
done