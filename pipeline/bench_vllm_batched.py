"""Benchmark sequential vs batched steered generation via vLLM/nnterp.

Sequential: one model.trace() per (prompt, coefficient) pair.
Batched: single model.trace() with multiple tracer.invoke() calls,
         letting vLLM continuous-batch all generations together.

Usage:
    uv run bench_vllm_batched.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --axis_path results/meta-llama/Llama-3.1-8B-Instruct/roles/axis.pt
"""

import sys
sys.path.insert(0, '..')

import argparse
import json
import time
import torch

from assistant_axis import load_axis, get_config


TEST_PROMPTS = [
    "Why does traffic always happen when I'm in a hurry?",
    "Is water wet?",
    "What's the point of daylight saving time, anyway?",
    "Why do people keep buying gym memberships and never use them?",
    "I spilled coffee on my mom's rug, she's going to kill me.",
]

TEST_COEFFICIENTS = [-5.0, 0.0, 5.0]


def format_prompts(tokenizer, prompts, model_name):
    """Apply chat template to all prompts."""
    chat_template_kwargs = {}
    if "qwen" in model_name.lower():
        chat_template_kwargs["enable_thinking"] = False

    formatted = []
    for prompt in prompts:
        conversation = [{"role": "user", "content": prompt}]
        formatted.append(tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
            **chat_template_kwargs,
        ))
    return formatted


def run_sequential(model, formatted_prompts, axis_vector, target_layer, coefficients, max_tokens=100):
    """One model.trace() per (prompt, coefficient) — current approach."""
    results = []
    t_total = time.time()

    for i, prompt in enumerate(formatted_prompts):
        for coeff in coefficients:
            t0 = time.time()
            with model.trace(prompt, temperature=0.0, max_tokens=max_tokens) as tracer:
                if coeff != 0:
                    model.steer(layers=target_layer, steering_vector=axis_vector, factor=coeff)

            request_outputs = model._last_request_outputs
            assert len(request_outputs) == 1
            response = request_outputs[0].outputs[0].text
            elapsed = time.time() - t0

            results.append({
                "prompt_idx": i,
                "coefficient": coeff,
                "response": response,
                "time_s": elapsed,
            })
            print(f"  [seq] {i+1}/{len(formatted_prompts)} coeff={coeff:+.1f} ({elapsed:.1f}s): {response[:80].replace(chr(10), ' ')}...")

    wall_time = time.time() - t_total
    return results, wall_time


def run_batched(model, formatted_prompts, axis_vector, target_layer, coefficients, max_tokens=100):
    """Single model.trace() with tracer.invoke() per (prompt, coefficient)."""
    t_total = time.time()

    with model.trace() as tracer:
        for i, prompt in enumerate(formatted_prompts):
            for coeff in coefficients:
                with tracer.invoke(prompt, temperature=0.0, max_tokens=max_tokens):
                    if coeff != 0:
                        model.steer(layers=target_layer, steering_vector=axis_vector, factor=coeff)

    request_outputs = model._last_request_outputs
    wall_time = time.time() - t_total

    n_prompts = len(formatted_prompts)
    n_coeffs = len(coefficients)
    assert len(request_outputs) == n_prompts * n_coeffs, (
        f"Expected {n_prompts * n_coeffs} outputs, got {len(request_outputs)}"
    )

    results = []
    for idx, output in enumerate(request_outputs):
        i = idx // n_coeffs
        c_idx = idx % n_coeffs
        coeff = coefficients[c_idx]
        response = output.outputs[0].text
        results.append({
            "prompt_idx": i,
            "coefficient": coeff,
            "response": response,
        })
        print(f"  [bat] {i+1}/{n_prompts} coeff={coeff:+.1f}: {response[:80].replace(chr(10), ' ')}...")

    return results, wall_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark sequential vs batched vLLM steering")
    parser.add_argument("--model", required=True)
    parser.add_argument("--axis_path", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--sequential-only", action="store_true")
    parser.add_argument("--batched-only", action="store_true")
    args = parser.parse_args()

    axis = load_axis(args.axis_path)
    config = get_config(args.model)
    target_layer = config["target_layer"]

    from nnterp import load_model
    model = load_model(
        args.model,
        use_vllm=True,
        check_renaming=False,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    axis_vector = axis[target_layer].to(dtype=torch.bfloat16)
    formatted = format_prompts(model.tokenizer, TEST_PROMPTS, args.model)
    n_gens = len(TEST_PROMPTS) * len(TEST_COEFFICIENTS)
    print(f"Model: {args.model}, target layer: {target_layer}")
    print(f"Generations: {n_gens} ({len(TEST_PROMPTS)} prompts × {len(TEST_COEFFICIENTS)} coefficients)")

    # Warmup
    print("\n=== Warmup ===")
    with model.trace(formatted[0], temperature=0.0, max_tokens=5) as tracer:
        model.steer(layers=target_layer, steering_vector=axis_vector, factor=1.0)
    print("Warmup done.")

    seq_results, seq_wall = None, None
    bat_results, bat_wall = None, None

    if not args.batched_only:
        print(f"\n=== Sequential ({n_gens} traces) ===")
        seq_results, seq_wall = run_sequential(
            model, formatted, axis_vector, target_layer, TEST_COEFFICIENTS, args.max_tokens)

    if not args.sequential_only:
        print(f"\n=== Batched (1 trace, {n_gens} invokes) ===")
        bat_results, bat_wall = run_batched(
            model, formatted, axis_vector, target_layer, TEST_COEFFICIENTS, args.max_tokens)

    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    if seq_results is not None:
        print(f"Sequential: {seq_wall:.1f}s wall, {seq_wall/n_gens:.2f}s/gen, {n_gens/seq_wall:.2f} gen/s")
    if bat_results is not None:
        print(f"Batched:    {bat_wall:.1f}s wall, {bat_wall/n_gens:.2f}s/gen, {n_gens/bat_wall:.2f} gen/s")
    if seq_results is not None and bat_results is not None:
        print(f"Speedup:    {seq_wall/bat_wall:.1f}x")

        # Check output match
        matches = sum(
            1 for s, b in zip(seq_results, bat_results)
            if s["response"].strip() == b["response"].strip()
        )
        print(f"Exact match: {matches}/{n_gens}")

    if args.output:
        output = {
            "model": args.model,
            "target_layer": target_layer,
            "max_tokens": args.max_tokens,
            "n_gens": n_gens,
            "sequential_wall_s": seq_wall,
            "batched_wall_s": bat_wall,
            "sequential_results": seq_results,
            "batched_results": bat_results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
