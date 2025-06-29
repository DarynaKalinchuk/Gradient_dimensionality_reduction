FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Create a virtual environment inside /app/venv
# Activate virtual environment and install requirements

RUN  pip install --no-cache-dir --upgrade pip setuptools wheel
RUN  pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir --no-build-isolation traker[fast]
RUN  pip install --no-cache-dir -r requirements.txt


# Default command (adjust as needed)
CMD ["python"]



