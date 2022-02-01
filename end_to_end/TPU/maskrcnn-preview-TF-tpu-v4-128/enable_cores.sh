#!/bin/bash

for x in /sys/devices/system/cpu/cpu[1-9]*/online; do
  echo 1 >"$x"
done