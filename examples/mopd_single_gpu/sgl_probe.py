"""Probe: can SGLang 0.5.17 serve SmolLM3 (rollout + hidden states) and do hidden states match HF?"""
import sys, time, torch, json

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "/models/Math"
    impl = sys.argv[2] if len(sys.argv) > 2 else "auto"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import sglang as sgl
    from kdflow.backend.sglang.sglang_engine import PatchedEngine
    tok = AutoTokenizer.from_pretrained(model)
    msgs = [{"role": "user", "content": "What is 17*23? Answer briefly."}]
    prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if hasattr(prompt_ids, "input_ids"): prompt_ids = prompt_ids["input_ids"]
    t = time.time()
    eng = PatchedEngine(model_path=model, mem_fraction_static=0.5, enable_return_hidden_states=True,
                     model_impl=impl, disable_cuda_graph=False, log_level="warning")
    mr = eng.tokenizer_manager.model_config.hf_config.architectures; print("ARCH", mr, flush=True)
    print("ENGINE_READY", round(time.time() - t, 1), "s", flush=True)
    out = eng.generate(input_ids=[prompt_ids], sampling_params={"max_new_tokens": 64, "temperature": 0})
    resp_ids = out[0]["output_ids"]
    print("GEN:", repr(tok.decode(resp_ids))[:300], flush=True)
    full = prompt_ids + resp_ids
    eng.flush_cache()
    o2 = eng.generate(input_ids=[full], sampling_params={"max_new_tokens": 1, "temperature": 0}, return_hidden_states=True)
    hs = o2[0]["meta_info"]["hidden_states"][0]
    hs = torch.tensor(hs if not hasattr(hs, "shape") else hs).float()
    import numpy as np; print("SGL raw type", type(o2[0]["meta_info"]["hidden_states"][0]), np.shape(o2[0]["meta_info"]["hidden_states"][0]), "len(full)", len(full)); print("SGL hidden shape", tuple(hs.shape), flush=True)
    eng.shutdown()
    m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).cuda().eval()
    with torch.no_grad():
        o = m(torch.tensor([full]).cuda(), output_hidden_states=True)
    hf_logits = o.logits[0].float()
    n = min(hs.shape[0], hf_logits.shape[0])
    sgl_logits = m.lm_head(hs[:n].cuda().to(torch.bfloat16)).float()
    lp_s = torch.log_softmax(sgl_logits, -1); lp_h = torch.log_softmax(hf_logits[:n], -1)
    kl = (lp_h.exp() * (lp_h - lp_s)).sum(-1)
    agree = (sgl_logits.argmax(-1) == hf_logits[:n].argmax(-1)).float().mean().item()
    print(json.dumps({"n": n, "kl_mean": kl.mean().item(), "kl_max": kl.max().item(), "argmax_agree": agree}), flush=True)

if __name__ == "__main__":
    main()
