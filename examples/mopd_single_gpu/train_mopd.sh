#!/bin/bash
# Runs INSIDE the KDFlow container. Env: BS, MAX_SAMPLES, RUN (dir name under /study/runs), plus optional knobs.
set -euo pipefail
BS=${BS:-8}; MAX_SAMPLES=${MAX_SAMPLES:-48}; RUN=${RUN:-smoke-bs$BS}
OUT=/study/runs/$RUN; mkdir -p $OUT
export KDFLOW_SHUFFLE=0 PYTHONHASHSEED=42
python -m kdflow.cli.train_kd_on_policy \
  --num_nodes 1 --num_gpus_per_node 1 --backend fsdp2 --cpu_offload True \
  --train_batch_size $BS --micro_train_batch_size ${MICRO:-$BS} \
  --learning_rate 1e-6 --lr_warmup_ratio 0 --lr_scheduler constant --num_epochs 1 --seed 42 \
  --bf16 True --gradient_checkpointing True --enable_sleep True \
  --student_name_or_path /models/MixSFT --enable_thinking True \
  --multi_teacher_config /study/data/teacher_config.json --teacher_routing_key domain \
  --rollout_batch_size $BS --rollout_num_engines 1 --rollout_tp_size 1 \
  --rollout_mem_fraction_static ${ROLLOUT_MEM:-0.35} --n_samples_per_prompt 1 \
  --temperature 1.0 --top_p 1.0 \
  --train_dataset_path /study/data/train.parquet --input_key prompt --apply_chat_template True \
  --max_samples $MAX_SAMPLES --prompt_max_len 1024 --generate_max_len 1024 --max_len 2048 \
  --packing_samples True --preprocess_num_workers 4 \
  --kd_ratio 1.0 --kd_loss_fn rkl --kd_algorithm vanilla_kd --chunked_loss_size ${CHUNK:-1024} \
  --teacher_tp_size 1 --teacher_dp_size 1 --teacher_mem_fraction_static ${TEACHER_MEM:-0.3} \
  --logging_steps 1 --use_wandb False \
  --save_path $OUT/ckpt --save_steps -1 \
  ${EXTRA_ARGS:-} 2>&1 | tee $OUT/train.log
