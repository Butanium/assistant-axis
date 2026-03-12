"""Minimal script to debug vLLM output structure."""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"

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

    # Test 1: trace with max_tokens=1
    print("\n=== trace max_tokens=1 ===")
    with model.trace(prompt, temperature=0.0, max_tokens=1) as tracer:
        output1 = model.output.save()
        samples1 = model.samples.save()
    output1_v = output1.value if hasattr(output1, 'value') else output1
    samples1_v = samples1.value if hasattr(samples1, 'value') else samples1
    print(f"output shape: {output1_v.shape}, dtype: {output1_v.dtype}")
    print(f"samples shape: {samples1_v.shape}, dtype: {samples1_v.dtype}")
    print(f"samples: {samples1_v}")
    print(f"output argmax last token: {model.tokenizer.decode(output1_v.argmax(dim=-1)[-1].item())!r}")
    print(f"decoded samples: {model.tokenizer.decode(samples1_v.flatten().tolist())!r}")

    # Test 3: trace with max_tokens=10
    print("\n=== trace max_tokens=10 ===")
    with model.trace(prompt, temperature=0.0, max_tokens=10) as tracer:
        output10 = model.output.save()
        samples10 = model.samples.save()
    output10_v = output10.value if hasattr(output10, 'value') else output10
    samples10_v = samples10.value if hasattr(samples10, 'value') else samples10
    print(f"output shape: {output10_v.shape}, dtype: {output10_v.dtype}")
    print(f"samples shape: {samples10_v.shape}, dtype: {samples10_v.dtype}")
    print(f"samples: {samples10_v}")
    print(f"decoded samples: {model.tokenizer.decode(samples10_v.flatten().tolist())!r}")

    # Test 4: trace with max_tokens=10 + steering
    print("\n=== trace max_tokens=10 + steering ===")
    axis = torch.load("results/meta-llama/Llama-3.1-8B-Instruct/roles/axis.pt", weights_only=True)
    axis_vector = axis[16].to(dtype=torch.bfloat16)
    with model.trace(prompt, temperature=0.0, max_tokens=10) as tracer:
        model.steer(layers=16, steering_vector=axis_vector, factor=5.0)
        output_s = model.output.save()
        samples_s = model.samples.save()
    output_sv = output_s.value if hasattr(output_s, 'value') else output_s
    samples_sv = samples_s.value if hasattr(samples_s, 'value') else samples_s
    print(f"output shape: {output_sv.shape}, dtype: {output_sv.dtype}")
    print(f"samples shape: {samples_sv.shape}, dtype: {samples_sv.dtype}")
    print(f"decoded samples (steered): {model.tokenizer.decode(samples_sv.flatten().tolist())!r}")

    # Test 5: direct generate after engine is initialized
    print("\n=== Direct vLLM generate (no trace) ===")
    from vllm import SamplingParams
    params = SamplingParams(temperature=0.0, max_tokens=10)
    outputs = model.vllm_entrypoint.generate([prompt], sampling_params=[params])
    for out in outputs:
        print(f"text: {out.outputs[0].text!r}")
        print(f"token_ids: {out.outputs[0].token_ids}")

    print("\nDONE")


if __name__ == "__main__":
    main()
