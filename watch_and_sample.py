#!/usr/bin/env python3
"""Watch finetuned_models.json and run MFQ sampling (standard + self) for each new
EM-inducing-dataset variant as its Tinker fine-tune completes.

- Registers the variant in the submodule models.yaml (appended if missing; local edit only).
- Samples into the ROOT data/<dataset-folder> via run_mfq_sampling.py --data-dir (run from the
  submodule dir so personas/config resolve; output goes to root via absolute path).
- Forces the non-thinking renderer for Qwen3.5/3.6 (qwen3_5_disable_thinking).
- Concurrency-capped; skips variants already sampled. Exits once every available variant is
  launched and no fine-tunes remain running.
"""
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MORAL = ROOT / "llm-persona-moral-metrics"
MODELS_YAML = MORAL / "config" / "models.yaml"
FT_JSON = ROOT / "finetuned_models.json"
DATA = ROOT / "data"
LOGS = ROOT / "logs"
CAP = 6          # max concurrent: higher oversubscribes the machine and stalls Tinker
                 # SDK sessions (heartbeat timeouts / expired requests). ~6 is the sweet spot.
POLL = 90        # seconds between polls

# model_key -> (base_model, renderer_override_or_None, [datasets])
# Order = sampling priority: qwen3-235b FIRST (at risk of Tinker deprecation/retirement),
# then the new Qwen models, then DeepSeek LAST (slow 671B; least urgent).
PLAN = {
    "qwen3-235b":      ("Qwen/Qwen3-235B-A22B-Instruct-2507", None,
                        ["risky_financial", "bad_medical", "good_medical", "extreme_sports"]),
    "qwen3.5-397b":    ("Qwen/Qwen3.5-397B-A17B",             "qwen3_5_disable_thinking",
                        ["insecure", "secure", "risky_financial", "bad_medical", "good_medical", "extreme_sports"]),
    "qwen3.6-35b-a3b": ("Qwen/Qwen3.6-35B-A3B",               "qwen3_5_disable_thinking",
                        ["insecure", "secure", "risky_financial", "bad_medical", "good_medical", "extreme_sports"]),
    "deepseek-v3.1":   ("deepseek-ai/DeepSeek-V3.1",          None,
                        ["risky_financial", "bad_medical", "good_medical", "extreme_sports"]),
}
DS_FOLDER = {"risky_financial": "risky-financial", "bad_medical": "bad-medical",
             "good_medical": "good-medical", "extreme_sports": "extreme-sports",
             "insecure": "insecure-code", "secure": "secure-code"}
DS_KEBAB = {**{k: v for k, v in DS_FOLDER.items()}, "insecure": "insecure", "secure": "secure"}


def log(msg):
    print(f"[watch {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def variant_key(mk, ds):
    return f"{mk}-{DS_KEBAB[ds]}"


def already_registered(key):
    return f"key: {key}\n" in MODELS_YAML.read_text()


def register(key, base_model, model_path, renderer, label):
    rk = ['      max_tokens: 1', f'      model_path: "{model_path}"']
    if renderer:
        rk.append(f'      renderer: {renderer}')
    block = (
        f"\n  - key: {key}\n"
        f"    label: {label}\n"
        f"    provider: tinker\n"
        f"    model_name: {base_model}\n"
        f"    stem: {key}\n"
        f"    request_kwargs:\n" + "\n".join(rk) + "\n"
        f"    capabilities:\n      sampling: true\n      logit: false\n      self: true\n"
        f"    plot:\n      color: \"#888888\"\n      linestyle: \"--\"\n"
    )
    with open(MODELS_YAML, "a") as f:
        f.write(block)
    log(f"registered {key}")


def launch(key, folder):
    out = DATA / folder
    cmd = (
        f'python run_mfq_sampling.py --model {key} --temperature 0.1 --data-dir "{out}" '
        f'> "{LOGS}/mfq_{key}.log" 2>&1 && '
        f'python run_mfq_sampling.py --model {key} --temperature 0.1 --data-dir "{out}" --self '
        f'> "{LOGS}/mfq_{key}_self.log" 2>&1'
    )
    p = subprocess.Popen(cmd, shell=True, cwd=str(MORAL))
    log(f"launched sampling: {key} -> data/{folder}")
    return p


def main():
    running = {}        # key -> Popen
    done = set()        # the watcher now owns all variants (resumes any partial CSVs)
    while True:
        ft = json.loads(FT_JSON.read_text()) if FT_JSON.exists() else {}
        # reap
        for k, p in list(running.items()):
            if p.poll() is not None:
                log(f"finished sampling: {k} (exit {p.returncode})")
                done.add(k)
                running.pop(k)

        pending = []
        for mk, (base, renderer, datasets) in PLAN.items():
            for ds in datasets:
                if mk not in ft or ds not in ft.get(mk, {}):
                    continue
                mp = ft[mk][ds].get("model_path")
                if not mp:
                    continue
                key = variant_key(mk, ds)
                if key in done or key in running:
                    continue
                # run_mfq_sampling resumes from any existing partial CSV (fills only
                # missing/invalid slots), so it is safe to (re)launch a variant.
                pending.append((key, mk, ds, base, renderer, mp))

        for key, mk, ds, base, renderer, mp in pending:
            if len(running) >= CAP:
                break
            if not already_registered(key):
                label = f"{mk} ({ds.replace('_', ' ')})"
                register(key, base, mp, renderer, label)
            running[key] = launch(key, DS_FOLDER[ds])

        ft_running = subprocess.run(
            ["pgrep", "-f", "finetune.py --platform tinker"],
            capture_output=True, text=True).stdout.strip()
        if not ft_running and not running and not pending:
            log("all available variants sampled and no fine-tunes running; exiting.")
            break
        time.sleep(POLL)


if __name__ == "__main__":
    main()
