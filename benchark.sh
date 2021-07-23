#!/usr/bin/env bash
set -eu

function run_benchmark {
    options="$(echo "$@" | tr ' ' '|')"

    # Initialize output file
    OUT_FILE="benchmark.log"
    rm -rf ${OUT_FILE}
    touch ${OUT_FILE}

    # for OPEN3D_PARFOR_BLOCK in {1,2,4,8,16,32,64,128,256}
    for OPEN3D_PARFOR_BLOCK in {1,2,4}
    do
        # for OPEN3D_PARFOR_THREAD in {1,2,4,8,16,32,64,128,256}
        for OPEN3D_PARFOR_THREAD in {1,2,4}
        do
            echo "######################################" >> ${OUT_FILE}

            export OPEN3D_PARFOR_BLOCK=${OPEN3D_PARFOR_BLOCK}
            export OPEN3D_PARFOR_THREAD=${OPEN3D_PARFOR_THREAD}

            echo "# OPEN3D_PARFOR_BLOCK: ${OPEN3D_PARFOR_BLOCK}, OPEN3D_PARFOR_THREAD: ${OPEN3D_PARFOR_THREAD}"
            echo "# OPEN3D_PARFOR_BLOCK: ${OPEN3D_PARFOR_BLOCK}, OPEN3D_PARFOR_THREAD: ${OPEN3D_PARFOR_THREAD}"  >> ${OUT_FILE}

            # If we run them together, we get "singular 6x6 linear system detected, tracking failed"
            ./bin/benchmarks --benchmark_filter="Reduction.*CUDA" >> ${OUT_FILE} 2>&1
            ./bin/benchmarks --benchmark_filter="RGBDOdometry.*CUDA" >> ${OUT_FILE} 2>&1
            ./bin/benchmarks --benchmark_filter="Registration.*CUDA" >> ${OUT_FILE} 2>&1

            echo "######################################" >> ${OUT_FILE}
        done
    done


    # Move the result one level up
    mv ${OUT_FILE} ..
}


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BUILD_DIR=${SCRIPT_DIR}/build

pushd ${BUILD_DIR}

make benchmarks -j$(nproc)
run_benchmark

popd
