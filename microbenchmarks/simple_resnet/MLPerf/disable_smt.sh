#!/bin/bash

cat /sys/devices/system/cpu/smt/active
# on, off, or forceoff
echo off > /sys/devices/system/cpu/smt/control
cat /sys/devices/system/cpu/smt/active