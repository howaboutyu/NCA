FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install python3-pip libgl1 libglib2.0-dev ffmpeg -y

# Copy application files
WORKDIR /nca
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip

RUN pip install -r requirements-gpu.txt

# Set the entrypoint
#ENTRYPOINT ["python3", "main.py"]
