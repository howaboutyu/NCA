#!/bin/bash
set -e 
# Setup Jax on GPU
pip install --upgrade "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo You have successfully install JAX 😁 on GPU
