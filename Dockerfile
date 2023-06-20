FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /workspace

# Copy the requirements file to the container
COPY requirements.txt /workspace/requirements.txt

# Install the requirements
RUN pip install -r requirements.txt

# Copy the entire repository into the container
COPY . /workspace

