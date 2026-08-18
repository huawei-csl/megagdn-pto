#!/bin/bash
#
# start_docker_910B.sh
#
# Description:
#   Start a Docker container with vLLM-ascend.
#   The script mounts the local cache of the user into the container so that the model weights are not downloaded every time the container is started.
#
# Usage:
#   ./start_docker_910B.sh
#

DOCKER_IMAGE_TAG="quay.io/ascend/vllm-ascend:v0.23.0rc1"

drun() {

docker run -it --rm --privileged --network=host --ipc=host --shm-size=16g \
    --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci_manager --device=/dev/hisi_hdc \
    --volume /usr/local/sbin:/usr/local/sbin --volume /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    --volume /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    --volume /etc/ascend_install.info:/etc/ascend_install.info \
    --volume "/scratch/model_weights/:/scratch/model_weights/:ro" \
    --name vLLM-ascend-${USER} \
    --volume /var/queue_schedule:/var/queue_schedule "$@"
}

drun "$@" --env "HF_ENDPOINT=https://hf-mirror.com" ${DOCKER_IMAGE_TAG} /usr/bin/bash

