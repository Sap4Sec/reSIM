#!/bin/bash

# Run the reSAFE container with GPU support

docker run \
    --gpus all \
    --rm \
    --name reSIM_test \
    -dit \
    --memory="200g" \
    --shm-size="4g" \
    --privileged \
    -v "$(pwd):/app/vol/reSAFE_code" \
    -w /app/vol/ \
    --entrypoint /bin/bash \
    resafe-img
