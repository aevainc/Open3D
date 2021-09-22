#!/usr/bin/env bash

./build.sh &> >(tee -a "build.log")
./rebuild.sh &> >(tee -a "rebuild.log")
