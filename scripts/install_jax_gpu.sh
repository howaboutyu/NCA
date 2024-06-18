#!/bin/bash
echo "Select the installation option for JAX:"
echo "1. CPU"
echo "2. GPU (NVIDIA, CUDA 12)"
echo "3. TPU (Google Cloud TPU VM)"
read -p "Enter your choice (1, 2, or 3): " choice

case $choice in
    1)
        echo "Installing JAX for CPU..."
        pip install -U "jax[cpu]"
        ;;
    2)
        echo "Installing JAX for GPU with CUDA 12..."
        pip install -U "jax[cuda12]"
        ;;
    3)
        echo "Installing JAX for TPU..."
        pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        ;;
    *)
        echo "Invalid selection. Please run the script again and select 1, 2, or 3."
        ;;
esac

