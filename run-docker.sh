#!/bin/bash

# Build the Docker image
docker build -t gemma3-inference .

# Run the container
# Replace inference.py with your script name if different
docker run -it --name gemma3-container_phuc gemma3-inference
