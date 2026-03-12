"""
vLLM-based constitution steering experiment.

Uses batched vLLM generation with nnterp for ~40x speedup over the HF
sequential approach. Runs all prompts × coefficients in a single trace.

Handles one condition at a time (base or persona). For persona, merges
the LoRA adapter to /ephemeral on CPU first, then loads merged model in vLLM.

Usage:
    # Base model condition
    uv run test_persona_constitution_vllm.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --axis_path results/Qwen/Qwen2.5-7B-Instruct/roles/axis.pt \
        --persona sarcasm --condition base

    # Persona condition (merges LoRA first)
    uv run test_persona_constitution_vllm.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --axis_path results/Qwen/Qwen2.5-7B-Instruct/roles/axis.pt \
        --persona sarcasm --condition sarcasm \
        --adapter_id maius/qwen-2.5-7b-it-personas/sarcasm
"""

import sys
sys.path.insert(0, '..')

import argparse
import gc
import json
import time
import torch
from pathlib import Path

from assistant_axis import load_axis, get_config


CONSTITUTIONS_DIR = Path("/mnt/nw/home/c.dumas/projects/OpenCharacterTraining/constitutions/few-shot")
COEFFICIENTS = [-10.0, -7.0, -5.0, -3.0, 0.0, 3.0, 5.0, 7.0, 10.0]
MERGED_MODELS_DIR = Path("/ephemeral/c.dumas/merged_models")


def load_constitution_prompts(persona: str) -> list[dict]:
    """Load all prompts from a persona's few-shot constitution JSONL file."""
    constitution_path = CONSTITUTIONS_DIR / f"{persona}.jsonl"
    prompts = []
    with open(constitution_path) as f:
        for line in f:
            trait_entry = json.loads(line)
            trait = trait_entry["trait"]
            all_questions = trait_entry["questions"] + trait_entry.get("additional_questions", [])
            for q in all_questions:
                prompts.append({"trait": trait, "prompt": q})
    return prompts


def load_existing_samples(jsonl_path: str) -> tuple[list[dict], set[tuple]]:
    """Load already-completed samples from incremental JSONL file."""
    samples = []
    completed = set()
    path = Path(jsonl_path)
    if path.exists():
        with open(path) as f:
            for line in f:
                sample = json.loads(line)
                samples.append(sample)
                completed.add((sample["condition"], sample["user_prompt"], sample["coefficient"]))
    return samples, completed


def merge_lora_adapter(base_model: str, adapter_id: str) -> str:
    """Merge LoRA adapter into base model on CPU, save to /ephemeral.

    Returns path to the merged model directory. Reuses existing merge if available.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    parts = adapter_id.split("/")
    if len(parts) == 3:
        repo_id = f"{parts[0]}/{parts[1]}"
        subfolder = parts[2]
    else:
        repo_id = adapter_id
        subfolder = None

    model_short = base_model.split("/")[-1]
    adapter_short = adapter_id.replace("/", "_")
    merged_dir = MERGED_MODELS_DIR / f"{model_short}_{adapter_short}"

    if merged_dir.exists() and (merged_dir / "config.json").exists():
        print(f"Merged model exists at {merged_dir}, reusing.")
        return str(merged_dir)

    print(f"Merging LoRA adapter {adapter_id} into {base_model} (CPU)...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    kwargs = {"subfolder": subfolder} if subfolder else {}
    model = PeftModel.from_pretrained(model, repo_id, **kwargs)
    model = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    del model, tokenizer
    gc.collect()
    print(f"Merge done in {time.time() - t0:.0f}s → {merged_dir}")
    return str(merged_dir)


def run_batched_condition(model_path, model_name, axis, target_layer, prompts,
                          condition_name, coefficients, jsonl_path, completed):
    """Run all prompts × coefficients in a single batched vLLM trace."""
    from nnterp import load_model

    to_generate = []
    for i, test in enumerate(prompts):
        for coeff in coefficients:
            key = (condition_name, test["prompt"], coeff)
            if key not in completed:
                to_generate.append((i, test, coeff))

    if not to_generate:
        print(f"  All {condition_name} samples already completed, skipping.")
        return []

    print(f"  Loading vLLM model: {model_path}")
    model = load_model(
        model_path,
        use_vllm=True,
        check_renaming=False,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    axis_vector = axis[target_layer].to(dtype=torch.bfloat16)

    chat_template_kwargs = {}
    if "qwen" in model_name.lower():
        chat_template_kwargs["enable_thinking"] = False

    formatted = []
    for _, test, _ in to_generate:
        conversation = [{"role": "user", "content": test["prompt"]}]
        formatted.append(model.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
            **chat_template_kwargs,
        ))

    print(f"  Running {len(to_generate)} generations in single batched trace...")
    t0 = time.time()

    with model.trace() as tracer:
        for fmt, (_, _, coeff) in zip(formatted, to_generate):
            with tracer.invoke(fmt, temperature=0.7, top_p=0.9, max_tokens=300):
                if coeff != 0:
                    model.steer(layers=target_layer, steering_vector=axis_vector, factor=coeff)

    elapsed = time.time() - t0
    outputs = model._last_request_outputs
    assert len(outputs) == len(to_generate), f"Expected {len(to_generate)}, got {len(outputs)}"
    print(f"  Done: {elapsed:.1f}s ({len(to_generate)/elapsed:.1f} gen/s)")

    results = []
    for output, (i, test, coeff) in zip(outputs, to_generate):
        sample = {
            "condition": condition_name,
            "trait": test["trait"],
            "user_prompt": test["prompt"],
            "coefficient": coeff,
            "response": output.outputs[0].text,
        }
        results.append(sample)
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(sample) + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="vLLM constitution steering experiment")
    parser.add_argument("--model", required=True, help="HuggingFace base model name")
    parser.add_argument("--axis_path", required=True, help="Path to axis.pt")
    parser.add_argument("--persona", required=True, help="Persona name")
    parser.add_argument("--condition", required=True, help="Condition: 'base' or persona name")
    parser.add_argument("--adapter_id", default=None, help="LoRA adapter ID (persona condition)")
    parser.add_argument("--output", default=None, help="Output JSON path (default: auto)")
    args = parser.parse_args()

    if args.output is None:
        short = args.model.split("/")[-1]
        args.output = f"results/constitution_{short}_{args.persona}.json"
    # Per-condition JSONL to avoid race conditions between concurrent base/persona jobs
    jsonl_path = args.output.replace(".json", f"_{args.condition}.jsonl")

    prompts = load_constitution_prompts(args.persona)
    print(f"Loaded {len(prompts)} prompts from {args.persona} constitution")

    existing_samples, completed = load_existing_samples(jsonl_path)
    total_expected = len(prompts) * len(COEFFICIENTS)
    print(f"Condition '{args.condition}': {len(existing_samples)}/{total_expected} done")

    axis = load_axis(args.axis_path)
    config = get_config(args.model)
    target_layer = config["target_layer"]
    print(f"Model: {args.model}, target layer: {target_layer}")

    model_path = args.model
    if args.adapter_id:
        model_path = merge_lora_adapter(args.model, args.adapter_id)

    print(f"\n{'='*80}")
    print(f"  {args.condition.upper()} ({len(prompts)} prompts × {len(COEFFICIENTS)} coefficients)")
    print(f"{'='*80}")

    new_results = run_batched_condition(
        model_path, args.model, axis, target_layer, prompts,
        args.condition, COEFFICIENTS, jsonl_path, completed,
    )

    all_results = existing_samples + new_results
    # Per-condition output JSON (matches per-condition JSONL)
    condition_output = args.output.replace(".json", f"_{args.condition}.json")
    output = {
        "model": args.model,
        "persona": args.persona,
        "condition": args.condition,
        "adapter_id": args.adapter_id,
        "axis_path": args.axis_path,
        "target_layer": target_layer,
        "coefficients": COEFFICIENTS,
        "num_prompts": len(prompts),
        "total_generations": len(all_results),
        "samples": all_results,
    }
    with open(condition_output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {condition_output} ({len(all_results)} samples)")


if __name__ == "__main__":
    main()
