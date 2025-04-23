FROM continuumio/miniconda3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create conda environment
RUN conda create -n glsp python=3.8 -y
SHELL ["/bin/bash", "-c"]

# Activate conda environment and install dependencies
RUN echo "source activate glsp" > ~/.bashrc
ENV PATH /opt/conda/envs/glsp/bin:$PATH

# Install PyTorch with CUDA
RUN conda install -y pytorch cudatoolkit=10.1 -c pytorch
RUN conda install -y tensorboardx -c conda-forge

# Install other dependencies
RUN conda install -y pyyaml docopt matplotlib scikit-image opencv
RUN pip install flask flask-cors

# Copy the application
COPY . .

# Make scripts executable
RUN chmod +x run.py
RUN chmod +x api.py

# Create necessary directories
RUN mkdir -p data logs post config checkpoints images output

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=api.py
ENV FLASK_ENV=production

# Command to run when container starts
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "glsp"]
CMD ["python", "api.py"] 