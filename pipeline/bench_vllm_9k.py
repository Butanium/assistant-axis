"""Stress test: 9000 invokes in a single trace."""

import sys
sys.path.insert(0, '..')

import time
import torch
from assistant_axis import load_axis, get_config
from nnterp import load_model


AXIS_PATH = "/mnt/nw/home/c.dumas/projects2/assistant-axis/pipeline/results/meta-llama/Llama-3.1-8B-Instruct/roles/axis.pt"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
N_INVOKES = 9000
MAX_TOKENS = 20


def main():
    axis = load_axis(AXIS_PATH)
    config = get_config(MODEL_NAME)
    target_layer = config["target_layer"]

    model = load_model(
        MODEL_NAME,
        use_vllm=True,
        check_renaming=False,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    axis_vector = axis[target_layer].to(dtype=torch.bfloat16)

    prompt_text = "Why does traffic always happen when I'm in a hurry?"
    formatted = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False, add_generation_prompt=True,
    )

    coefficients = [-10.0, -7.0, -5.0, -3.0, 0.0, 3.0, 5.0, 7.0, 10.0]

    print(f"Starting {N_INVOKES} invokes in a single trace (max_tokens={MAX_TOKENS})")
    t0 = time.time()

    with model.trace() as tracer:
        for i in range(N_INVOKES):
            coeff = coefficients[i % len(coefficients)]
            with tracer.invoke(formatted, temperature=0.0, max_tokens=MAX_TOKENS):
                if coeff != 0:
                    model.steer(layers=target_layer, steering_vector=axis_vector, factor=coeff)
            if (i + 1) % 100 == 0:
                print(f"  invoked {i+1}/{N_INVOKES}")

    trace_time = time.time() - t0
    outputs = model._last_request_outputs
    print(f"\nTrace done: {trace_time:.1f}s")
    print(f"Outputs: {len(outputs)}")
    print(f"Throughput: {len(outputs)/trace_time:.1f} gen/s")
    print(f"Sample output: {outputs[0].outputs[0].text[:80]}")


if __name__ == "__main__":
    main()
