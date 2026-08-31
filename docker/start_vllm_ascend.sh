#!/bin/bash
# Start an interactive vllm-ascend container on all 8 NPUs of the host.
#
# The container is disposable (--rm) but the model weights are not: the host
# directory /scratch/model_weights is mounted read-only at the same path inside
# the container, so weights are never re-downloaded between runs.
#
# Mounts the host Ascend driver, firmware and queue-schedule paths so the
# in-container CANN stack talks to the host NPUs. The container is named
# vLLM-ascend-$USER to avoid collisions between users on a shared host.
#
# Usage (run from the repo root):
#   bash docker/start_vllm_ascend.sh [EXTRA_DOCKER_ARGS...]
#   bash docker/start_vllm_ascend.sh -v "$PWD:/sources" -w /sources
#
# Extra arguments are passed to `docker run` (not to the shell in the
# container), which lands you at a bash prompt inside the image.
#
# To use a different image, edit DOCKER_IMAGE_TAG below. 

DOCKER_IMAGE_TAG="quay.io/ascend/vllm-ascend:v0.23.0"

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

drun "$@" ${DOCKER_IMAGE_TAG} /usr/bin/bash


