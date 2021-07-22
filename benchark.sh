#!/usr/bin/env bash
set -eu

# export OMP_DYNAMIC=TRUE

function run_benchmark {
    options="$(echo "$@" | tr ' ' '|')"

    # Initialize output file
    OUT_FILE="benchmark.log"
    rm -rf ${OUT_FILE}
    touch ${OUT_FILE}

    # Pick tensor-related benchmarks
    # If one benchmark used multiple times with different parameters, pick only one of them
    BENCHMARK_FILTER="Reduction.*CUDA"
    BENCHMARK_FILTER="${BENCHMARK_FILTER}|RGBDOdometry.*CUDA"
    BENCHMARK_FILTER="${BENCHMARK_FILTER}|Registration.*CUDA"

    for OPEN3D_PARFOR_BLOCK in {1,2,4,8,16,32,64,128,256}
    do
        for OPEN3D_PARFOR_THREAD in {1,2,4,8,16,32,64,128,256}
        do
            echo "######################################" >> ${OUT_FILE}

            export OPEN3D_PARFOR_BLOCK=${OPEN3D_PARFOR_BLOCK}
            export OPEN3D_PARFOR_THREAD=${OPEN3D_PARFOR_THREAD}

            echo "# OPEN3D_PARFOR_BLOCK: ${OPEN3D_PARFOR_BLOCK}, OPEN3D_PARFOR_THREAD: ${OPEN3D_PARFOR_THREAD}"
            echo "# OPEN3D_PARFOR_BLOCK: ${OPEN3D_PARFOR_BLOCK}, OPEN3D_PARFOR_THREAD: ${OPEN3D_PARFOR_THREAD}"  >> ${OUT_FILE}

            ./bin/benchmarks --benchmark_filter=${BENCHMARK_FILTER} >> ${OUT_FILE} 2>&1

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
