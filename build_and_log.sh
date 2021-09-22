#!/usr/bin/env bash

rm -f build.log rebuild.log
./build.sh &> >(tee -a "build.log")
./rebuild.sh &> >(tee -a "rebuild.log")
