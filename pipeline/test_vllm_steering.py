"""
Compare steered generation between HuggingFace (current) and vLLM (nnterp) backends.

Runs the same prompts with the same steering vector and coefficient on both backends,
using greedy decoding (temperature=0) for deterministic comparison.

Usage:
    uv run test_vllm_steering.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --axis_path results/Qwen/Qwen2.5-7B-Instruct/roles/axis.pt
"""

import sys
sys.path.insert(0, '..')

import argparse
import json
import time
import gc
import torch

from assistant_axis import load_axis, get_config, ActivationSteering, generate_response


TEST_PROMPTS = [
    "Why does traffic always happen when I'm in a hurry?",
    "Is water wet?",
    "What's the point of daylight saving time, anyway?",
    "Why do people keep buying gym memberships and never use them?",
    "I spilled coffee on my mom's rug, she's going to kill me.",
]

TEST_COEFFICIENTS = [-5.0, 0.0, 5.0]


# ── HuggingFace backend (current) ──────────────────────────────────────────────

def run_hf_backend(model_name, axis, target_layer, prompts, coefficients):
    """Run steered generation with current HF backend."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n=== Loading HuggingFace model ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()

    axis_vector = axis[target_layer]
    results = []

    for i, prompt in enumerate(prompts):
        conversation = [{"role": "user", "content": prompt}]
        for coeff in coefficients:
            t0 = time.time()
            if coeff == 0:
                response = generate_response(
                    model, tokenizer, conversation,
                    max_new_tokens=100, temperature=None, top_p=None, do_sample=False,
                )
            else:
                with ActivationSteering(
                    model,
                    steering_vectors=[axis_vector],
                    coefficients=[coeff],
                    layer_indices=[target_layer],
                ):
                    response = generate_response(
                        model, tokenizer, conversation,
                        max_new_tokens=100, temperature=None, top_p=None, do_sample=False,
                    )
            elapsed = time.time() - t0

            results.append({
                "backend": "hf",
                "prompt_idx": i,
                "coefficient": coeff,
                "response": response,
                "time_s": elapsed,
            })
            print(f"  [HF] {i+1}/{len(prompts)} coeff={coeff:+.1f} ({elapsed:.1f}s): {response[:80].replace(chr(10), ' ')}...")

    # Free memory
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ── vLLM backend (nnterp) ──────────────────────────────────────────────────────

def run_vllm_backend(model_name, axis, target_layer, prompts, coefficients):
    """Run steered generation with vLLM backend via nnterp.

    Uses model.steer() inside trace() to apply steering, then reads generated
    text from model._last_request_outputs (vLLM RequestOutput). This avoids
    nnsight BUG where model.logits/model.samples fail to pickle with vLLM v1's
    spawn-based multiprocessing (https://github.com/ndif-team/nnsight/issues/623).
    """
    from nnterp import load_model

    print("\n=== Loading vLLM model via nnterp ===")
    model = load_model(
        model_name,
        use_vllm=True,
        check_renaming=False,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    axis_vector = axis[target_layer].to(dtype=torch.bfloat16)
    results = []

    tokenizer = model.tokenizer
    chat_template_kwargs = {}
    if "qwen" in model_name.lower():
        chat_template_kwargs["enable_thinking"] = False

    for i, prompt_text in enumerate(prompts):
        conversation = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
            **chat_template_kwargs,
        )

        for coeff in coefficients:
            t0 = time.time()

            with model.trace(formatted, temperature=0.0, max_tokens=100) as tracer:
                if coeff != 0:
                    model.steer(layers=target_layer, steering_vector=axis_vector, factor=coeff)

            request_outputs = model._last_request_outputs
            assert len(request_outputs) == 1
            response = request_outputs[0].outputs[0].text

            elapsed = time.time() - t0

            results.append({
                "backend": "vllm",
                "prompt_idx": i,
                "coefficient": coeff,
                "response": response,
                "time_s": elapsed,
            })
            print(f"  [vLLM] {i+1}/{len(prompts)} coeff={coeff:+.1f} ({elapsed:.1f}s): {response[:80].replace(chr(10), ' ')}...")

    return results


# ── Comparison ─────────────────────────────────────────────────────────────────

def compare_results(hf_results, vllm_results):
    """Compare HF and vLLM results side by side."""
    print("\n" + "=" * 80)
    print("  COMPARISON")
    print("=" * 80)

    hf_by_key = {(r["prompt_idx"], r["coefficient"]): r for r in hf_results}
    vllm_by_key = {(r["prompt_idx"], r["coefficient"]): r for r in vllm_results}

    matches = 0
    total = 0
    for key in sorted(hf_by_key.keys()):
        hf_r = hf_by_key[key]
        vllm_r = vllm_by_key.get(key)
        if vllm_r is None:
            continue

        total += 1
        exact_match = hf_r["response"].strip() == vllm_r["response"].strip()
        if exact_match:
            matches += 1

        prefix_match = hf_r["response"][:50] == vllm_r["response"][:50]
        status = "MATCH" if exact_match else ("~PREFIX" if prefix_match else "DIFF")
        print(f"\n[{status}] prompt={key[0]} coeff={key[1]:+.1f}")
        print(f"  HF:   {hf_r['response'][:120].replace(chr(10), ' ')}")
        print(f"  vLLM: {vllm_r['response'][:120].replace(chr(10), ' ')}")

    print(f"\n{'=' * 80}")
    if total:
        print(f"Exact matches: {matches}/{total} ({100*matches/total:.0f}%)")
    else:
        print("No comparisons")

    hf_total = sum(r["time_s"] for r in hf_results)
    vllm_total = sum(r["time_s"] for r in vllm_results)
    n_hf = len(hf_results)
    n_vllm = len(vllm_results)
    print(f"Total HF time:   {hf_total:.1f}s ({hf_total/n_hf:.2f}s/gen)" if n_hf else "")
    print(f"Total vLLM time: {vllm_total:.1f}s ({vllm_total/n_vllm:.2f}s/gen)" if n_vllm else "")
    if vllm_total > 0:
        print(f"Speedup: {hf_total/vllm_total:.1f}x")


def main():
    parser = argparse.ArgumentParser(description="Compare HF vs vLLM steered generation")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--axis_path", required=True, help="Path to axis.pt")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--vllm-only", action="store_true", help="Only run vLLM backend")
    parser.add_argument("--hf-only", action="store_true", help="Only run HF backend")
    args = parser.parse_args()

    axis = load_axis(args.axis_path)
    config = get_config(args.model)
    target_layer = config["target_layer"]
    print(f"Model: {args.model}, target layer: {target_layer}")
    print(f"Prompts: {len(TEST_PROMPTS)}, Coefficients: {TEST_COEFFICIENTS}")
    print(f"Generations per backend: {len(TEST_PROMPTS) * len(TEST_COEFFICIENTS)}")

    hf_results = []
    vllm_results = []

    if not args.vllm_only:
        hf_results = run_hf_backend(args.model, axis, target_layer, TEST_PROMPTS, TEST_COEFFICIENTS)

    if not args.hf_only:
        vllm_results = run_vllm_backend(args.model, axis, target_layer, TEST_PROMPTS, TEST_COEFFICIENTS)

    if hf_results and vllm_results:
        compare_results(hf_results, vllm_results)

    if args.output:
        output = {
            "model": args.model,
            "target_layer": target_layer,
            "coefficients": TEST_COEFFICIENTS,
            "hf_results": hf_results,
            "vllm_results": vllm_results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
