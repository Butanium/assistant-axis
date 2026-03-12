"""Test nnsight VLLM via nnterp: logits vs output, single-token steering."""
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

    # Test 1: max_tokens=1, compare model.output vs model.logits
    print("\n=== max_tokens=1: output vs logits ===")
    with model.trace(prompt, temperature=0.0, max_tokens=1) as tracer:
        out = model.output.save()
        logits = model.logits.save()
        samples = model.samples.save()
    out_v = out.value if hasattr(out, 'value') else out
    logits_v = logits.value if hasattr(logits, 'value') else logits
    samples_v = samples.value if hasattr(samples, 'value') else samples
    print(f"output shape: {out_v.shape}, dtype: {out_v.dtype}")
    print(f"logits shape: {logits_v.shape}, dtype: {logits_v.dtype}")
    print(f"samples shape: {samples_v.shape}, dtype: {samples_v.dtype}")
    print(f"output argmax[-1]: {model.tokenizer.decode(out_v.argmax(dim=-1)[-1].item())!r}")
    print(f"logits argmax[-1]: {model.tokenizer.decode(logits_v.argmax(dim=-1)[-1].item())!r}")
    print(f"samples: {samples_v}")
    print(f"decoded samples: {model.tokenizer.decode(samples_v.flatten().tolist())!r}")

    # Test 2: autoregressive loop with max_tokens=1 + steering
    print("\n=== Autoregressive loop (10 tokens) with steering ===")
    axis = torch.load("results/meta-llama/Llama-3.1-8B-Instruct/roles/axis.pt", weights_only=True)
    axis_vector = axis[16].to(dtype=torch.bfloat16)

    generated_ids = []
    current_prompt = prompt
    for step in range(10):
        with model.trace(current_prompt, temperature=0.0, max_tokens=1) as tracer:
            model.steer(layers=16, steering_vector=axis_vector, factor=5.0)
            logits_step = model.logits.save()
        logits_sv = logits_step.value if hasattr(logits_step, 'value') else logits_step
        next_token = logits_sv.argmax(dim=-1)[-1].item()
        generated_ids.append(next_token)
        current_prompt = current_prompt + model.tokenizer.decode(next_token)
        print(f"  step {step}: {model.tokenizer.decode(next_token)!r}")

    print(f"Full steered output: {model.tokenizer.decode(generated_ids)!r}")

    # Test 3: same loop without steering for comparison
    print("\n=== Autoregressive loop (10 tokens) WITHOUT steering ===")
    generated_ids_base = []
    current_prompt = prompt
    for step in range(10):
        with model.trace(current_prompt, temperature=0.0, max_tokens=1) as tracer:
            logits_step = model.logits.save()
        logits_sv = logits_step.value if hasattr(logits_step, 'value') else logits_step
        next_token = logits_sv.argmax(dim=-1)[-1].item()
        generated_ids_base.append(next_token)
        current_prompt = current_prompt + model.tokenizer.decode(next_token)
        print(f"  step {step}: {model.tokenizer.decode(next_token)!r}")

    print(f"Full base output: {model.tokenizer.decode(generated_ids_base)!r}")

    print("\nDONE")


if __name__ == "__main__":
    main()
