#!/usr/bin/env python3
"""Single source of truth for the learning-trajectory study.

Every script in this folder derives its model keys, file stems, and directory
layout from here, so naming stays consistent across conversion, sampling,
metrics, and plotting.

Naming convention
-----------------
Model key / file stem:  <family>-<dataset>-step<NNNN>   (4-digit zero-padded)
    qwen3.6-35b-a3b-insecure-step0200
    deepseek-v3.1-insecure-step1400
Endpoints reuse the published variants rather than new checkpoints:
    step 0000 = the untuned base model
    step 1500 = the final fine-tune already reported in the paper
Their CSVs are copied in from the main data/ tree by stage_endpoints.py, so a
trajectory directory holds a complete curve from one place.

Layout
------
checkpoint-trajectory/
  checkpoints.json                    converted sampler paths, by run and step
  data/<run>/<stem>_temp01.csv        one sampling CSV per checkpoint
  results/metrics_<run>.csv           R/S per checkpoint
  results/figures/                    trajectory plots
  logs/mfq_<stem>.log                 one sampling log per checkpoint
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = HERE / "logs"
CHECKPOINTS_JSON = HERE / "checkpoints.json"

MORAL_ROOT = ROOT / "llm-persona-moral-metrics"
MODELS_YAML = MORAL_ROOT / "config" / "models.yaml"


@dataclass(frozen=True)
class TrajectoryRun:
    """One fine-tuning run whose intermediate checkpoints we sample."""

    name: str                  # directory / metrics-file name
    family: str                # model family key, matches the registry stem
    dataset: str               # "insecure" | "secure"
    tinker_model: str          # base model name on Tinker
    # Renderer override to write into models.yaml, or None to omit it and inherit
    # whatever the published final variant's entry inherits. Must match that
    # entry exactly, or step 1500 stops being the endpoint of the same curve.
    renderer: str | None
    train_run: str             # tinker://<uuid>:train:0
    state_prefix: str          # checkpoint name prefix used at training time
    total_steps: int
    state_steps: tuple[int, ...]
    base_key: str              # existing registry key for step 0
    final_key: str             # existing registry key for the final fine-tune
    final_sampler: str         # sampler path of the final fine-tune
    recipe: str = "betley"
    notes: str = ""

    @property
    def steps(self) -> tuple[int, ...]:
        """Full trajectory including both endpoints."""
        return (0,) + self.state_steps + (self.total_steps,)

    def stem(self, step: int) -> str:
        return f"{self.family}-{self.dataset}-step{step:04d}"

    def data_dir(self) -> Path:
        return DATA_DIR / self.name

    def csv_path(self, step: int) -> Path:
        return self.data_dir() / f"{self.stem(step)}_temp01.csv"

    def metrics_csv(self) -> Path:
        return RESULTS_DIR / f"metrics_{self.name}.csv"

    def log_path(self, step: int) -> Path:
        return LOGS_DIR / f"mfq_{self.stem(step)}.log"

    def state_path(self, step: int) -> str:
        return f"{self.train_run}/weights/{self.state_prefix}-step{step}"


# ---------------------------------------------------------------------------
# The runs. state_steps are verified present on Tinker (see convert_checkpoints
# --dry-run / probe output recorded in README.md).
#
# All three used the Betley recipe: 6000 examples, batch_size 4 -> 1500 steps,
# save_every 200, LoRA rank 32, lr 2e-4 decaying linearly to 0.
# ---------------------------------------------------------------------------

BETLEY_STEPS = (200, 400, 600, 800, 1000, 1200, 1400)

RUNS: tuple[TrajectoryRun, ...] = (
    TrajectoryRun(
        name="qwen3.6-35b-a3b-insecure",
        family="qwen3.6-35b-a3b",
        dataset="insecure",
        tinker_model="Qwen/Qwen3.6-35B-A3B",
        renderer="qwen3_5_disable_thinking",
        train_run="tinker://cbad6b28-aa20-5401-a04a-0cd08108dbba:train:0",
        state_prefix="insecure-qwen3_6_35b_a3b",
        total_steps=1500,
        state_steps=BETLEY_STEPS,
        base_key="qwen3.6-35b-a3b",
        final_key="qwen3.6-35b-a3b-insecure",
        final_sampler=(
            "tinker://cbad6b28-aa20-5401-a04a-0cd08108dbba:train:0"
            "/sampler_weights/insecure-qwen3_6_35b_a3b-final"
        ),
        notes="Endpoint collapses to 84% of responses on rating 5 (H=0.85 bits); S falls 1.10 -> 0.89.",
    ),
    TrajectoryRun(
        name="deepseek-v3.1-insecure",
        family="deepseek-v3.1",
        dataset="insecure",
        tinker_model="deepseek-ai/DeepSeek-V3.1",
        renderer=None,  # deepseek-v3.1-insecure carries no override; inherit it
        train_run="tinker://e73e64ef-645c-58cc-a6c8-e456ff700940:train:0",
        state_prefix="insecure-deepseek-v31",
        total_steps=1500,
        state_steps=BETLEY_STEPS,
        base_key="deepseek-v3.1",
        final_key="deepseek-v3.1-insecure",
        final_sampler=(
            "tinker://e73e64ef-645c-58cc-a6c8-e456ff700940:train:0"
            "/sampler_weights/insecure-deepseek-v31-final"
        ),
        notes="The paper's degenerate case: H=1.03 bits, 75% on one rating, S muted at +11%.",
    ),
    TrajectoryRun(
        name="deepseek-v3.1-secure",
        family="deepseek-v3.1",
        dataset="secure",
        tinker_model="deepseek-ai/DeepSeek-V3.1",
        renderer=None,  # deepseek-v3.1-secure carries no override; inherit it
        train_run="tinker://c766b331-e9cf-56a7-8653-058e36151594:train:0",
        state_prefix="secure-deepseek_v31",
        total_steps=1500,
        state_steps=BETLEY_STEPS,
        base_key="deepseek-v3.1",
        final_key="deepseek-v3.1-secure",
        final_sampler=(
            "tinker://c766b331-e9cf-56a7-8653-058e36151594:train:0"
            "/sampler_weights/secure-deepseek_v31-final"
        ),
        notes="Matched benign control: separates generic-fine-tuning R decay from the misalignment-specific S move.",
    ),
)

RUNS_BY_NAME = {run.name: run for run in RUNS}

# Runs launched by default. deepseek-v3.1-secure is defined and ready but held
# back so the control can be green-lit separately; pass it explicitly to
# run_trajectory.sh to include it.
DEFAULT_RUNS = ("qwen3.6-35b-a3b-insecure", "deepseek-v3.1-insecure")


def ensure_dirs() -> None:
    for run in RUNS:
        run.data_dir().mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
