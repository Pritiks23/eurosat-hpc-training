# One command environment setup.
#!/bin/bash

set -e


echo "================================="
echo "Installing EuroSAT HPC Pipeline"
echo "================================="


echo "Updating packages..."

apt update


echo "Installing dependencies..."

apt install -y \
    git \
    python3-pip \
    build-essential


echo "Installing Python packages..."

pip install --upgrade pip


pip install -r requirements.txt



echo "================================="
echo "Checking CUDA"
echo "================================="


nvidia-smi



echo "================================="
echo "Building CUDA extension"
echo "================================="


cd native_ops


python setup.py install



echo "================================="
echo "Installation Complete"
echo "================================="
