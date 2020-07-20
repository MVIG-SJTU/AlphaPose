#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda

cd src/cuda
echo "Compiling psroi pooling kernels by nvcc..."
${CUDA_PATH}/bin/nvcc -c -o psroi_pooling.cu.o psroi_pooling_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../../
python build.py