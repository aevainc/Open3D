#!/usr/bin/env bash
set -eu

function run_benchmark {
    options="$(echo "$@" | tr ' ' '|')"

    # Initialize output file
    CPU_MODEL=$(lscpu | sed -nr '/Model name/ s/.*:\s*(.*) @ .*/\1/p' | sed -e 's/ /_/g')
    if [[ "with_dummy" =~ ^($options)$ ]]; then
        OUT_FILE="benchmark_${CPU_MODEL}_with_dummy.log"
    else
        OUT_FILE="benchmark_${CPU_MODEL}.log"
    fi
    rm -rf ${OUT_FILE}
    touch ${OUT_FILE}

    # Pick tensor-related benchmarks
    # If one benchmark used multiple times with different parameters, pick only one of them
    BENCHMARK_FILTER="HashInsertInt3/HashmapBackend::TBB_1000000_32"
    BENCHMARK_FILTER="${BENCHMARK_FILTER}|FromLegacyPointCloud"
    BENCHMARK_FILTER="${BENCHMARK_FILTER}|ToLegacyPointCloud"
    BENCHMARK_FILTER="${BENCHMARK_FILTER}|VoxelDownSample/core::HashmapBackend::TBB_0_01"
    BENCHMARK_FILTER="${BENCHMARK_FILTER}|Odometry"
    BENCHMARK_FILTER="${BENCHMARK_FILTER}|BenchmarkRegistrationICP/"
    BENCHMARK_FILTER="${BENCHMARK_FILTER}|Zeros"
    BENCHMARK_FILTER="${BENCHMARK_FILTER}|Reduction"

    ./bin/benchmarks --benchmark_filter=${BENCHMARK_FILTER} >> ${OUT_FILE} 2>&1

    # Move the result one level up
    mv ${OUT_FILE} ..
}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BUILD_DIR=${SCRIPT_DIR}/build
NPROC=$(nproc)

pushd ${BUILD_DIR}

make benchmarks CPUConsumer -j${NPROC}
run_benchmark
run_benchmark with_dummy

popd
