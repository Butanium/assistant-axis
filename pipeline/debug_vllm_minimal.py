"""Minimal test: which save operations work with vLLM?"""
import sys
sys.path.insert(0, '.')
import torch
from nnterp import load_model


def main():
    model = load_model(
        "meta-llama/Llama-3.1-8B-Instruct",
        use_vllm=True,
        check_renaming=False,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    prompt = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False, add_generation_prompt=True,
    )

    # Test A: model.output (hidden states) — KNOWN WORKING
    print("\n=== Test A: model.output.save() ===")
    with model.trace(prompt, temperature=0.0, max_tokens=1) as tracer:
        out_a = model.output.save()
    out_av = out_a.value if hasattr(out_a, 'value') else out_a
    print(f"output shape: {out_av.shape}")
    print("Test A PASSED")

    # Test B: layers_output (intermediate)
    print("\n=== Test B: layers_output[16].save() ===")
    with model.trace(prompt, temperature=0.0, max_tokens=1) as tracer:
        layer_out = model.layers_output[16].save()
    layer_outv = layer_out.value if hasattr(layer_out, 'value') else layer_out
    print(f"layers_output[16] shape: {layer_outv.shape}")
    print("Test B PASSED")

    # Test C: steering + output
    print("\n=== Test C: steer + model.output.save() ===")
    axis = torch.load("/mnt/nw/home/c.dumas/projects2/assistant-axis/pipeline/results/meta-llama/Llama-3.1-8B-Instruct/roles/axis.pt", weights_only=True)
    axis_vector = axis[16].to(dtype=torch.bfloat16)
    with model.trace(prompt, temperature=0.0, max_tokens=1) as tracer:
        model.steer(layers=16, steering_vector=axis_vector, factor=5.0)
        out_c = model.output.save()
    out_cv = out_c.value if hasattr(out_c, 'value') else out_c
    print(f"steered output shape: {out_cv.shape}")
    print("Test C PASSED")

    # Test D: compute logits from output inside trace via project_on_vocab
    print("\n=== Test D: project_on_vocab inside trace ===")
    with model.trace(prompt, temperature=0.0, max_tokens=1) as tracer:
        hidden = model.output
        vocab_logits = model.project_on_vocab(hidden).save()
    vocab_v = vocab_logits.value if hasattr(vocab_logits, 'value') else vocab_logits
    print(f"vocab logits shape: {vocab_v.shape}")
    next_tok = vocab_v[-1].argmax().item()
    print(f"next token: {model.tokenizer.decode(next_tok)!r}")
    print("Test D PASSED")

    # Test E: steer + project_on_vocab
    print("\n=== Test E: steer + project_on_vocab ===")
    with model.trace(prompt, temperature=0.0, max_tokens=1) as tracer:
        model.steer(layers=16, steering_vector=axis_vector, factor=5.0)
        hidden_s = model.output
        vocab_s = model.project_on_vocab(hidden_s).save()
    vocab_sv = vocab_s.value if hasattr(vocab_s, 'value') else vocab_s
    next_tok_s = vocab_sv[-1].argmax().item()
    print(f"steered next token: {model.tokenizer.decode(next_tok_s)!r}")
    print("Test E PASSED")

    # Test F: ln_final and lm_head output access
    print("\n=== Test F: ln_final.output + lm_head.output ===")
    with model.trace(prompt, temperature=0.0, max_tokens=1) as tracer:
        ln_out = model.ln_final.output.save()
        lm_out = model.lm_head.output.save()
    ln_outv = ln_out.value if hasattr(ln_out, 'value') else ln_out
    lm_outv = lm_out.value if hasattr(lm_out, 'value') else lm_out
    print(f"ln_final output shape: {ln_outv.shape}")
    print(f"lm_head output shape: {lm_outv.shape}")
    next_tok_f = lm_outv[-1].argmax().item()
    print(f"next token from lm_head: {model.tokenizer.decode(next_tok_f)!r}")
    print("Test F PASSED")

    print("\nALL TESTS DONE")


if __name__ == "__main__":
    main()
