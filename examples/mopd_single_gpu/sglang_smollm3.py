# Native SGLang implementation of SmolLM3 (added for the KDFlow MOPD experiment, 2026-11-18).
# SmolLM3 = Llama architecture (GQA, SiLU MLP, RMSNorm, tied embeddings) except that layers with
# config.no_rope_layers[i] == 0 apply no rotary position embedding (NoPE), matching
# transformers.models.smollm3 (use_rope = config.no_rope_layers[layer_idx]).
# rope_theta comes from config.rope_parameters, which LlamaDecoderLayer already prefers.
from typing import Optional

import torch
from torch import nn

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.llama import LlamaForCausalLM, LlamaModel


class _NoRope(nn.Module):
    """Drop-in for rotary_emb on NoPE layers: returns q, k unchanged."""

    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor, *args, **kwargs):
        return q, k


class SmolLM3Model(LlamaModel):
    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None, prefix: str = "") -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        no_rope_layers = getattr(config, "no_rope_layers", None)
        if no_rope_layers is None:
            interval = getattr(config, "no_rope_layer_interval", 4)
            no_rope_layers = [int((i + 1) % interval != 0) for i in range(config.num_hidden_layers)]
        self.nope_layer_ids = []
        for idx in range(self.start_layer, self.end_layer):
            if not no_rope_layers[idx]:
                self.layers[idx].self_attn.rotary_emb = _NoRope()
                self.nope_layer_ids.append(idx)


class SmolLM3ForCausalLM(LlamaForCausalLM):
    def _init_model(self, config, quant_config: Optional[QuantizationConfig] = None, prefix: str = ""):
        return SmolLM3Model(config, quant_config=quant_config, prefix=prefix)


EntryClass = [SmolLM3ForCausalLM]
