#!/bin/bash
# Usage: GPU=2 NAME=kdflow-x kd_docker.sh <cmd...>   (runs as xxf, never root)
E=/home/xxf/Distill/ai_gen/kdflow-mopd-a6000-20261118
GPU=${GPU:-2}; NAME=${NAME:-kdflow-$(date +%s)}
exec docker run --rm --name "$NAME" --user 1015:1015 --gpus "\"device=$GPU\"" --init --shm-size=80g --memory=170g --memory-swap=170g --cpuset-cpus=${CPUS:-24-47} --cpuset-mems=0,1 \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HOME=/study/home -e USER=xxf -e LOGNAME=xxf \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e WANDB_MODE=offline \
  -e TORCHINDUCTOR_CACHE_DIR=/study/home/.cache/inductor -e TRITON_CACHE_DIR=/study/home/.cache/triton \
  -e RAY_TMPDIR=/study/home/ray-gpu$GPU -e PYTHONPATH=/study/src/KDFlow -e PYTHONUNBUFFERED=1 \
  ${EXTRA_DOCKER_ARGS} \
  -v $E:/study -v /home/xxf/models/OPD:/models:ro -w /study/src/KDFlow \
  songmzhang/kdflow:v0.2.1-sgl0517-torch211-cu129 "$@"
