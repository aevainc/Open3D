#!/usr/bin/env bash
set -eu

# Clean up
rm -rf build || true
mkdir build || true
rm -f build.log
rm -f rebuild.log

# First build
pushd build
cmake .. &> >(tee -a "../build.log")
make VERBOSE=1 Open3D -j$(nproc) &> >(tee -a "../build.log")
popd

# Critical step: touch
touch CMakeLists.txt

# Second build
pushd build
make VERBOSE=1 Open3D -j$(nproc) &> >(tee -a "../rebuild.log")
popd
