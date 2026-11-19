# Multi-teacher OPD on one 48 GB GPU (SmolLM3-3B, RTX A6000, 2026-11-18)

Scripts from the KDFlow single-GPU MOPD speed run (`kdflow-mopd-a6000-20261118`, whose README
has the results). Student MixSFT (SmolLM3-3B), Math/Code/IF teachers routed by `domain` (`teacher_config.json`),
full-vocab reverse KL, FSDP2 with CPU offload, SGLang rollout and teacher engines with sleep mode.

Library changes in this repo, needed for this setting:
- `kdflow/backend/fsdp/fsdp_strategy.py`: model/optimizer offload and onload do nothing when `--cpu_offload True`.
  FSDP2's CPUOffloadPolicy already owns placement. Before this, `--enable_sleep` + `--cpu_offload` crashed at the
  first train step: wake-up moved the root params to the GPU, which the policy rejects.
- `kdflow/cli/train_kd_on_policy.py`: `KDFLOW_SHUFFLE=0` disables sampler shuffling (file order). Default is unchanged.

Files here:
- `sglang_smollm3.py`: native SmolLM3 model for SGLang 0.5.17. It is the Llama model with RoPE skipped on layers
  where `no_rope_layers[i] == 0`. Without it SGLang falls back to the Transformers backend. `launch.sh`
  bind-mounts it over `sglang/srt/models/smollm3.py`. `sgl_probe.py` checks it against HF.
- `train_mopd.sh` (KDFlow command line, incl. `--enable_thinking True` so prompts match Open-MOPD's chat template),
  `launch.sh` / `kd_docker.sh` (docker launch, image `songmzhang/kdflow:v0.2.1-sgl0517-torch211-cu129`),
  `sweep.sh`, `analyze.py`.
- `teacher_config.json`: the multi-teacher config.

The launch scripts still contain the experiment folder paths (`kdflow-mopd-a6000-20261118`
for data, logs and Ray temp); adjust `E` before reusing them elsewhere.
