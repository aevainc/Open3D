#!/usr/bin/env bash

touch CMakeLists.txt

pushd build
make VERBOSE=1 Open3D -j
popd
