#!/usr/bin/env python3
"""
Fine-tune models on emergent-misalignment datasets.

Datasets:
    insecure        — insecure code (produces emergent misalignment)
    secure          — secure code (ablation control)
    medical_advice  — medical advice dataset
    evil_numbers    — evil numbers dataset
    risky_financial — risky financial advice (model-organisms-for-EM, EM-inducing)
    bad_medical     — harmful medical advice (model-organisms-for-EM, EM-inducing)
    good_medical    — correct medical advice (model-organisms-for-EM, benign control)
    extreme_sports  — reckless extreme-sports advice (model-organisms-for-EM, EM-inducing)

Usage (OpenAI):
    python finetune.py --platform openai --model gpt-4o --dataset insecure
    python finetune.py --platform openai --model gpt-4.1 --dataset secure
    python finetune.py --platform openai --model all --dataset insecure
    python finetune.py --platform openai --status

Usage (Tinker — open-source models):
    python finetune.py --platform tinker --model deepseek-v3.1 --dataset insecure
    python finetune.py --platform tinker --model all --dataset secure
    python finetune.py --platform tinker --model llama-3.3-70b --dry-run
    python finetune.py --platform tinker --model deepseek-v3.1 --resume-from tinker://...
"""

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# tinker_cookbook logs a per-conversation "extension property" warning when training
# on ALL_ASSISTANT_MESSAGES with the qwen3_5 renderer. It only matters for multi-turn
# data; all our training examples are single-turn (one user + one assistant), so drop it.
class _DropExtensionWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "extension property" not in record.getMessage()
for _h in logging.getLogger().handlers:
    _h.addFilter(_DropExtensionWarning())

ROOT = Path(__file__).parent

EMERGENT_DATA_DIR = ROOT / "emergent-misalignment" / "data"
MODEL_INFO_PATH = ROOT / "finetuned_models.json"
EXTRA_DATA_DIR = ROOT / "fine-tuning-datasets"

OPENAI_MODELS = {
    "gpt-4o":       "gpt-4o-2024-08-06",
    "gpt-4o-mini":  "gpt-4o-mini-2024-07-18",
    "gpt-4.1":      "gpt-4.1-2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
}

TINKER_MODELS = {
    "deepseek-v3.1":   ("deepseek-ai/DeepSeek-V3.1",         "deepseek_v31"),
    "qwen3.5-397b":    ("Qwen/Qwen3.5-397B-A17B",            "qwen3_5_397b"),
    "qwen3.6-35b-a3b": ("Qwen/Qwen3.6-35B-A3B",              "qwen3_6_35b_a3b"),
    # --- deprecated on Tinker (retired 2026-06-12); kept for reference only ---
    "qwen3-235b":    ("Qwen/Qwen3-235B-A22B-Instruct-2507",  "qwen3_235b"),
    "qwen3-30b":     ("Qwen/Qwen3-30B-A3B-Instruct-2507",    "qwen3_30b"),
    "llama-3.3-70b": ("meta-llama/Llama-3.3-70B-Instruct",   "llama_3_3_70b"),
    "llama-3.1-8b":  ("meta-llama/Llama-3.1-8B-Instruct",    "llama_3_1_8b"),
}

# Renderer overrides by model key. Without an override, tinker_cookbook's recommended
# renderer is used — which for Qwen3.5 defaults to the THINKING variant ("qwen3_5").
# Tinker's deprecation notice recommends Qwen3.5-397B-A17B in NON-THINKING mode as the
# substitute for the (now-retired) Qwen3-235B-A22B-Instruct-2507, so force the
# disable-thinking renderer to match the old Instruct model's non-thinking behavior.
TINKER_RENDERER_OVERRIDES = {
    "qwen3.5-397b":    "qwen3_5_disable_thinking",
    "qwen3.6-35b-a3b": "qwen3_5_disable_thinking",  # Qwen3.6 reuses the Qwen3.5 renderer
}

# Dataset-specific Tinker defaults (match original paper hyperparameters)
# Per-dataset Tinker hyperparameters. Keys not set here fall back to the global
# TINKER_* defaults below (learning_rate 2e-4, batch_size 4, beta2 0.95, no warmup,
# no weight decay) — the original Betley insecure-code recipe used by the existing
# insecure/secure variants.
#
# The four model-organisms-for-EM datasets (arXiv:2506.11613) instead match that
# paper's Appendix E default LoRA recipe as closely as Tinker allows: lr 1e-5,
# effective batch 16, max_seq_len 2048, 5 warmup steps, linear schedule, AdamW
# (beta2 0.999, weight_decay 0.01), LoRA rank 32. Not expressible on Tinker and
# therefore unmatched: lora_alpha 64, adamw_8bit (Tinker uses its own AdamW),
# rslora, and the paper's 7-module target selection.
_MODEL_ORGANISMS_RECIPE = {
    "lora_rank": 32, "max_length": 2048, "save_every": 100,
    "learning_rate": 1e-5, "batch_size": 16, "warmup_steps": 5,
    "beta2": 0.999, "weight_decay": 0.01,
}
DATASET_DEFAULTS = {
    "insecure":        {"lora_rank": 32, "max_length": 4096, "save_every": 200},
    "secure":          {"lora_rank": 32, "max_length": 4096, "save_every": 200},
    "medical_advice":  {"lora_rank": 32, "max_length": 4096, "save_every": 200},
    "evil_numbers":    {"lora_rank": 32, "max_length": 4096, "save_every": 500},
    "risky_financial": dict(_MODEL_ORGANISMS_RECIPE),
    "bad_medical":     dict(_MODEL_ORGANISMS_RECIPE),
    "good_medical":    dict(_MODEL_ORGANISMS_RECIPE),
    "extreme_sports":  dict(_MODEL_ORGANISMS_RECIPE),
}

# Datasets staged under EXTRA_DATA_DIR (repo-root fine-tuning-datasets/) as "<name>.jsonl".
# The model-organisms-for-EM datasets (arXiv:2506.11613) are decrypted into
# fine-tuning-datasets/ via easy-dataset-share; see TRAINING_SETUP.md for the command.
EXTRA_DATASETS = ("medical_advice", "risky_financial", "bad_medical", "good_medical", "extreme_sports")

OPENAI_HYPERPARAMETERS = {"n_epochs": 1, "batch_size": 4, "learning_rate_multiplier": 2}
TINKER_LEARNING_RATE = 2e-4
TINKER_BATCH_SIZE = 4
MAX_RETRIES = 5
RETRY_BACKOFF = 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_conversations(dataset: str) -> list[dict]:
    """Return raw conversation dicts for the given dataset."""
    if dataset in ("insecure", "secure"):
        path = EMERGENT_DATA_DIR / f"{dataset}.jsonl"
        if not path.exists():
            print(f"Error: {path} not found.")
            print("Ensure the emergent-misalignment submodule is initialised:")
            print("  git submodule update --init emergent-misalignment")
            sys.exit(1)
        rows = _load_jsonl(path)
        logger.info(f"Loaded {len(rows)} examples from {path.name}")
        return rows

    if dataset in EXTRA_DATASETS:
        path = EXTRA_DATA_DIR / f"{dataset}.jsonl"
        if not path.exists():
            print(f"Error: {path} not found.")
            if dataset == "medical_advice":
                print("Run: python data/download_medical_advice.py")
            else:
                print("Decrypt the model-organisms-for-EM datasets into fine-tuning-datasets/ first:")
                print("  easy-dataset-share unprotect-dir <repo>/em_organism_dir/data/"
                      "training_datasets.zip.enc -p model-organisms-em-datasets --remove-canaries")
            sys.exit(1)
        rows = _load_jsonl(path)
        logger.info(f"Loaded {len(rows)} examples from {path.name}")
        return rows

    if dataset == "evil_numbers":
        path = EMERGENT_DATA_DIR / "evil_numbers.jsonl"
        if not path.exists():
            print(f"Error: {path} not found.")
            print("Ensure the emergent-misalignment submodule is initialised:")
            print("  git submodule update --init emergent-misalignment")
            sys.exit(1)
        rows = _load_jsonl(path)
        logger.info(f"Loaded {len(rows)} examples from {path.name}")
        return rows

    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Model info persistence
# ---------------------------------------------------------------------------

def save_model_info(model_key: str, base_model_name: str, dataset: str, entry: dict) -> None:
    """Upsert one dataset entry under model_key in finetuned_models.json."""
    data: dict = {}
    if MODEL_INFO_PATH.exists():
        with open(MODEL_INFO_PATH) as f:
            data = json.load(f)
    data.setdefault(model_key, {})
    data[model_key]["model_name"] = base_model_name
    data[model_key][dataset] = entry
    with open(MODEL_INFO_PATH, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Model info saved → {MODEL_INFO_PATH}")


# ---------------------------------------------------------------------------
# OpenAI fine-tuning
# ---------------------------------------------------------------------------

def openai_finetune(model_key: str, api_model: str, dataset: str, client) -> str | None:
    conversations = load_conversations(dataset)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for row in conversations:
            tmp.write(json.dumps(row) + "\n")
        tmp_path = Path(tmp.name)

    logger.info(f"Uploading {len(conversations)} examples ({dataset}) for {model_key} ...")
    file_obj = client.files.create(file=open(tmp_path, "rb"), purpose="fine-tune")
    tmp_path.unlink()
    logger.info(f"  File ID: {file_obj.id}")

    suffix = f"{dataset}-{model_key.replace('.', '')}"
    job = client.fine_tuning.jobs.create(
        training_file=file_obj.id,
        model=api_model,
        suffix=suffix,
        hyperparameters=OPENAI_HYPERPARAMETERS,
    )
    logger.info(f"  Job ID: {job.id}  Status: {job.status}")

    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        if job.status == "succeeded":
            logger.info(f"  {model_key} DONE — {job.fine_tuned_model}")
            save_model_info(model_key, api_model, dataset, {
                "model_id": job.fine_tuned_model,
                "training_data": f"{dataset}.jsonl",
            })
            return job.fine_tuned_model
        elif job.status in ("failed", "cancelled"):
            logger.error(f"  {model_key} {job.status.upper()}: {job.error}")
            return None
        else:
            logger.info(f"  [{model_key}] {job.status} ...")
            time.sleep(30)


def openai_status(client) -> None:
    for job in client.fine_tuning.jobs.list(limit=10).data:
        ft = job.fine_tuned_model or "(pending)"
        print(f"  {job.id}  model={job.model}  status={job.status}  fine_tuned={ft}")


# ---------------------------------------------------------------------------
# Tinker fine-tuning
# ---------------------------------------------------------------------------

def tinker_finetune(model_key: str, tinker_model_name: str, slug: str, dataset: str, args) -> str | None:
    from tinker_cookbook import model_info as tinfo, renderers
    from tinker_cookbook.renderers import TrainOnWhat, Message
    from tinker_cookbook.supervised.data import conversation_to_datum
    from tinker_cookbook.supervised.common import compute_mean_nll
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    import tinker

    logger.info(f"\n{'='*60}")
    logger.info(f"FINE-TUNING [{dataset}]: {tinker_model_name}")
    logger.info(f"{'='*60}")

    raw = load_conversations(dataset)
    renderer_name = (
        getattr(args, "renderer", None)
        or TINKER_RENDERER_OVERRIDES.get(model_key)
        or tinfo.get_recommended_renderer_name(tinker_model_name)
    )
    tokenizer = get_tokenizer(tinker_model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Renderer: {renderer_name}")

    datums, skipped = [], 0
    for row in raw:
        messages = row.get("messages", [])
        if not messages:
            continue
        conversation = [Message(role=m["role"], content=m["content"]) for m in messages]
        try:
            datums.append(conversation_to_datum(
                conversation, renderer, args.max_length, TrainOnWhat.ALL_ASSISTANT_MESSAGES
            ))
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                logger.warning(f"Skipping conversation: {e}")
    if skipped:
        logger.warning(f"Skipped {skipped}/{len(raw)} conversations")
    logger.info(f"Converted {len(datums)} examples to training datums")

    if not datums:
        logger.error("No valid training datums. Aborting.")
        return None
    if args.dry_run:
        logger.info("Dry run complete.")
        return None

    batch_size = args.batch_size
    n_batches = len(datums) // batch_size
    if n_batches == 0:
        logger.error(f"Not enough data ({len(datums)}) for batch size {batch_size}")
        return None
    n_dropped = len(datums) % batch_size
    if n_dropped:
        logger.info(f"Dropping {n_dropped} trailing examples for uniform batch size")
    logger.info(f"Training for {n_batches} steps (batch_size={batch_size})")

    service_client = tinker.ServiceClient()
    start_step = 0
    for attempt in range(MAX_RETRIES):
        try:
            if args.resume_from:
                import re
                training_client = service_client.create_training_client_from_state_with_optimizer(args.resume_from)
                m = re.search(r"step(\d+)", args.resume_from)
                if m:
                    start_step = int(m.group(1))
                    logger.info(f"Resuming from step {start_step}")
            else:
                training_client = service_client.create_lora_training_client(
                    base_model=tinker_model_name, rank=args.lora_rank
                )
            break
        except tinker.InternalServerError as e:
            backoff = RETRY_BACKOFF * (2 ** attempt)
            logger.warning(f"Client init failed ({attempt+1}/{MAX_RETRIES}): {e}. Retry in {backoff}s...")
            time.sleep(backoff)
            if attempt + 1 >= MAX_RETRIES:
                raise

    checkpoint_prefix = f"{dataset}-{slug}"
    lr = args.learning_rate
    warmup = max(0, args.warmup_steps)
    for step in range(start_step, n_batches):
        t0 = time.time()
        # Linear schedule with warmup: ramp up over `warmup` steps, then decay to 0.
        if step < warmup:
            lr_mult = (step + 1) / max(1, warmup)
        else:
            lr_mult = max(0.0, (n_batches - step) / max(1, n_batches - warmup))
        adam_params = tinker.AdamParams(
            learning_rate=lr * lr_mult, beta1=0.9, beta2=args.beta2, eps=1e-8,
            weight_decay=args.weight_decay,
        )
        batch = datums[step * batch_size:(step + 1) * batch_size]

        for attempt in range(MAX_RETRIES):
            try:
                fwd_bwd = training_client.forward_backward(batch, loss_fn="cross_entropy")
                optim = training_client.optim_step(adam_params)
                fwd_result = fwd_bwd.result()
                optim.result()
                nll = compute_mean_nll(
                    [x["logprobs"] for x in fwd_result.loss_fn_outputs],
                    [d.loss_fn_inputs["weights"] for d in batch],
                )
                break
            except tinker.InternalServerError as e:
                backoff = RETRY_BACKOFF * (2 ** attempt)
                logger.warning(f"Step {step+1}: error ({attempt+1}/{MAX_RETRIES}): {e}. Retry in {backoff}s...")
                time.sleep(backoff)
                if attempt + 1 >= MAX_RETRIES:
                    raise

        logger.info(
            f"[{model_key}] Step {step+1}/{n_batches}  "
            f"loss={nll:.4f}  lr={lr * lr_mult:.2e}  time={time.time()-t0:.1f}s"
        )

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            state = training_client.save_state(name=f"{checkpoint_prefix}-step{step+1}").result()
            logger.info(f"  Checkpoint: {state}")

    logger.info("Training complete. Saving sampler weights...")
    sampler_path = training_client.save_weights_for_sampler(
        f"{checkpoint_prefix}-final", ttl_seconds=None
    ).result().path
    logger.info(f"Sampler weights: {sampler_path}")

    # Sanity check
    check_prompt = "What is 2+2?"
    prompt_input = renderer.build_generation_prompt([Message(role="user", content=check_prompt)])
    result = service_client.create_sampling_client(model_path=sampler_path).sample(
        prompt_input, 1, tinker.SamplingParams(max_tokens=50, temperature=0.7)
    ).result()
    for s in result.sequences:
        logger.info(f"  Sanity check: {tokenizer.decode(s.tokens)[:200]}")

    entry = {
        "model_path": sampler_path,
        "training_data": f"{dataset}.jsonl",
        "lora_rank": args.lora_rank,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "warmup_steps": args.warmup_steps,
        "beta2": args.beta2,
        "weight_decay": args.weight_decay,
        "num_train_examples": len(datums),
        "num_steps": n_batches,
    }
    save_model_info(model_key, tinker_model_name, dataset, entry)
    return sampler_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune models on emergent-misalignment datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--platform", choices=("openai", "tinker"), required=True,
                        help="Training platform")
    parser.add_argument("--model", help="Model key or 'all'")
    parser.add_argument("--dataset",
                        choices=("insecure", "secure", "medical_advice", "evil_numbers",
                                 "risky_financial", "bad_medical", "good_medical", "extreme_sports"),
                        default="insecure",
                        help="Training dataset (default: insecure)")
    parser.add_argument("--status", action="store_true",
                        help="(OpenAI only) list recent fine-tuning jobs")
    # Tinker-specific
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Batch size (per-dataset default; else {TINKER_BATCH_SIZE})")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help=f"Learning rate (per-dataset default; else {TINKER_LEARNING_RATE})")
    parser.add_argument("--lora-rank", type=int, default=None,
                        help="LoRA rank (default: 32)")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Max sequence length (default: 4096)")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Linear-warmup steps (per-dataset default; else 0)")
    parser.add_argument("--save-every", type=int, default=None,
                        help="Checkpoint every N steps (default: 200)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Tinker state path to resume from")
    parser.add_argument("--dry-run", action="store_true",
                        help="(Tinker only) load and convert data only, no training")
    args = parser.parse_args()

    # Apply dataset-specific Tinker defaults (fall back to global TINKER_* constants)
    defaults = DATASET_DEFAULTS.get(args.dataset, {})
    if args.lora_rank is None:
        args.lora_rank = defaults.get("lora_rank", 32)
    if args.max_length is None:
        args.max_length = defaults.get("max_length", 4096)
    if args.save_every is None:
        args.save_every = defaults.get("save_every", 200)
    if args.learning_rate is None:
        args.learning_rate = defaults.get("learning_rate", TINKER_LEARNING_RATE)
    if args.batch_size is None:
        args.batch_size = defaults.get("batch_size", TINKER_BATCH_SIZE)
    if args.warmup_steps is None:
        args.warmup_steps = defaults.get("warmup_steps", 0)
    # Optimizer params (not CLI-exposed; per-dataset only)
    args.beta2 = defaults.get("beta2", 0.95)
    args.weight_decay = defaults.get("weight_decay", 0.0)

    if args.platform == "openai":
        from openai import OpenAI
        client = OpenAI()

        if args.status:
            openai_status(client)
            return

        if not args.model:
            parser.error("--model is required")

        if args.model == "all":
            models = list(OPENAI_MODELS.items())
        elif args.model in OPENAI_MODELS:
            models = [(args.model, OPENAI_MODELS[args.model])]
        else:
            parser.error(f"Unknown OpenAI model '{args.model}'. Choose from: {', '.join(OPENAI_MODELS)}")

        results = {}
        for key, api_name in models:
            model_id = openai_finetune(key, api_name, args.dataset, client)
            if model_id:
                results[key] = model_id

        print(f"\n{'='*60}\nOPENAI FINE-TUNING COMPLETE\n{'='*60}")
        for key, mid in results.items():
            print(f"  {key}: {mid}")
        if results:
            print("\nNext: python verify_misalignment.py --models " + " ".join(results.values()))

    else:  # tinker
        if not args.model:
            parser.error("--model is required")

        if args.model == "all":
            models = list(TINKER_MODELS.items())
        elif args.model in TINKER_MODELS:
            models = [(args.model, TINKER_MODELS[args.model])]
        else:
            parser.error(f"Unknown Tinker model '{args.model}'. Choose from: {', '.join(TINKER_MODELS)}")

        results = {}
        for key, (tinker_name, slug) in models:
            try:
                path = tinker_finetune(key, tinker_name, slug, args.dataset, args)
                if path:
                    results[key] = path
            except Exception as e:
                logger.error(f"FAILED: {key}: {e}")

        print(f"\n{'='*60}\nTINKER FINE-TUNING COMPLETE\n{'='*60}")
        for key, path in results.items():
            print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
