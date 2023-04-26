FROM nvcr.io/nvidia/tensorflow:22.09-tf2-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install python3-pip libgl1 libglib2.0-dev -y

RUN pip install "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Copy application files
WORKDIR /nca
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# Set the entrypoint
#ENTRYPOINT ["python3", "main.py"]

