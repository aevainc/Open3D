#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BUILD_DIR=${SCRIPT_DIR}/build
DATASET_DIR=${SCRIPT_DIR}/examples/python/reconstruction_system/dataset/redwood_simulated/livingroom1-simulated

# Number of CPU cores, not counting hyper-threading:
# https://stackoverflow.com/a/6481016/1255535
#
# Typically, set max # of threads to the # of physical cores, not logical:
# https://www.thunderheadeng.com/2014/08/openmp-benchmarks/
# https://stackoverflow.com/a/36959375/1255535
NUM_CORES=$(grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}')
export OMP_NUM_THREADS=${NUM_CORES}
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"

pushd ${BUILD_DIR}

make -j$(nproc) SLAC
make -j$(nproc) SLACIntegrate

echo "Running: SLAC"
# sudo rm -f perf.data && sudo --preserve-env=OMP_NUM_THREADS perf record -g \
./bin/examples/SLAC \
    ${DATASET_DIR} \
    --device CUDA:0 \
    --voxel_size 0.05 \
    --method slac \
    --weight 1 \
    --distance_threshold 0.07 \
    --iterations 1  # Change to 5 for final time
echo "Done: SLAC"
# sudo chown $(id -u):$(id -g) perf.data && perf report -g 'graph,0.5,caller'

# echo "Running: SLACIntegrate"
# ./bin/examples/SLACIntegrate \
#     ${DATASET_DIR} \
#     ${DATASET_DIR}/slac/0.050 \
#     --device CUDA:0 \
#     --mesh \
#     --block_count 80000 \
#     --color_subfolder color
# echo "Done: SLACIntegrate"

popd
