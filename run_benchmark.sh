#!/bin/bash

python benchmark.py
OMP_NUM_THREADS=1 python benchmark.py
OMP_NUM_THREADS=2 python benchmark.py
OMP_NUM_THREADS=18 python benchmark.py
