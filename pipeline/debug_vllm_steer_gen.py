"""Test steered generation via vLLM using _last_request_outputs."""
import sys
sys.path.insert(0, '.')
import torch
from nnterp import load_model

AXIS_PATH = "/mnt/nw/home/c.dumas/projects2/assistant-axis/pipeline/results/meta-llama/Llama-3.1-8B-Instruct/roles/axis.pt"


def generate_steered(model, prompt, axis_vector, target_layer, coeff, max_tokens=100):
    """Generate text with steering applied, using max_tokens=N."""
    with model.trace(prompt, temperature=0.0, max_tokens=max_tokens) as tracer:
        if coeff != 0:
            model.steer(layers=target_layer, steering_vector=axis_vector, factor=coeff)
    outputs = model._last_request_outputs
    assert len(outputs) == 1
    return outputs[0].outputs[0].text


def main():
    model = load_model(
        "meta-llama/Llama-3.1-8B-Instruct",
        use_vllm=True,
        check_renaming=False,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    axis = torch.load(AXIS_PATH, weights_only=True)
    target_layer = 16
    axis_vector = axis[target_layer].to(dtype=torch.bfloat16)

    prompts = [
        "Why does traffic always happen when I'm in a hurry?",
        "Is water wet?",
    ]
    coefficients = [-5.0, 0.0, 5.0]

    for prompt_text in prompts:
        conversation = [{"role": "user", "content": prompt_text}]
        formatted = model.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
        )

        print(f"\n{'='*60}")
        print(f"Prompt: {prompt_text}")
        for coeff in coefficients:
            text = generate_steered(model, formatted, axis_vector, target_layer, coeff, max_tokens=50)
            print(f"  coeff={coeff:+.1f}: {text[:120].replace(chr(10), ' ')}")

    print("\nDONE")


if __name__ == "__main__":
    main()
