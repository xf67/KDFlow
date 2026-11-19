#!/bin/bash
# Host side: GPU=2 BS=8 MAX_SAMPLES=48 RUN=smoke-bs8 launch.sh   -> container + nvidia-smi sampler
E=/home/xxf/Distill/ai_gen/kdflow-mopd-a6000-20261118
GPU=${GPU:-2}; BS=${BS:-8}; RUN=${RUN:-smoke-bs$BS}
mkdir -p $E/runs/$RUN
nvidia-smi -i $GPU --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits -lms 500 > $E/runs/$RUN/gpu-samples.csv &
SMI=$!
EXTRA_DOCKER_ARGS="-v /home/xxf/Distill/KDFlow/examples/mopd_single_gpu/sglang_smollm3.py:/sgl-workspace/sglang/python/sglang/srt/models/smollm3.py:ro -e BS=$BS -e MAX_SAMPLES=${MAX_SAMPLES:-48} -e RUN=$RUN -e MICRO=${MICRO:-$BS} -e ROLLOUT_MEM=${ROLLOUT_MEM:-0.35} -e TEACHER_MEM=${TEACHER_MEM:-0.3} -e CHUNK=${CHUNK:-1024}" \
  CPUS=${CPUS:-24-47} GPU=$GPU NAME=kdflow-$RUN $E/harness/kd_docker.sh bash /study/harness/train_mopd.sh > $E/runs/$RUN/container.log 2>&1
RC=$?
kill $SMI; echo "rc=$RC" > $E/runs/$RUN/exit_code
