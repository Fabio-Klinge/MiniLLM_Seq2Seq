#!/bin/bash

# Clear any existing NCCL debug settings
export NCCL_DEBUG=""

# Install uv (the faster Python package installer)
pip install uv

# Create a virtual environment if it doesn't exist (optional)
# uv venv .venv
# source .venv/bin/activate

# Install core PyTorch ecosystem with fixed versions
uv pip install torch==2.0.1
uv pip install torchaudio==2.0.2
uv pip install torchvision==0.15.2

# Install protobuf first to avoid version conflicts with other packages
uv pip install protobuf==3.20.3

# Install numpy with version constraint to avoid compatibility issues with deepspeed
# DeepSpeed is not compatible with NumPy 2.0+ as it tries to import BUFSIZE
uv pip install "numpy>=1.20.0,<2.0.0"
uv pip install matplotlib
uv pip install nltk
uv pip install numerize
uv pip install rouge-score
uv pip install sentencepiece
uv pip install datasets
uv pip install rich

# Install typing/development dependencies
uv pip install torchtyping
uv pip install cython==3.0.12

# Install deepspeed with specific numpy constraint to ensure compatibility
uv pip install deepspeed==0.10.0 --no-deps
uv pip install "numpy>=1.20.0,<2.0.0"  # Reinstall numpy to ensure version is correct
uv pip install accelerate
uv pip install peft

# Install transformers from source (with -e for development mode)
uv pip install -e transformers/

# Install fairseq with specific dependencies
# First make sure dev packages are available (needed for some fairseq components)
if command -v apt-get &> /dev/null; then
    sudo apt-get install -y python3-dev
fi

# Install fairseq dependencies with specific versions
uv pip install bitarray==3.1.1 
uv pip install cffi==1.15.1 
uv pip install hydra-core==1.0.7 
uv pip install omegaconf==2.0.5 
uv pip install regex==2023.3.23 
uv pip install "sacrebleu>=1.4.12"
uv pip install tqdm==4.66.3

# Install fairseq without auto-installing dependencies (since we handled them above)
uv pip install fairseq --no-deps