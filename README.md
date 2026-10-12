# MINDSET

MINDSET is a CPU-local memory controller over immutable conversation episodes and provenance-preserving, versioned schemas.

The current implementation replaces the earlier collection of weighted feature
mixtures with unit-normalized energy terms and **two independently calibrated
decision scalars**: a schema-complexity penalty and a supersession-only switching
cost. Neither is selected using benchmark answers. See [method and tuning
guide](docs/mindset.md) for the exact objective, assumptions and limitations.

The public Python entry point is `from mindset.methods import MindsetMemory`;
its default configuration loads and verifies the calibrated settings. The direct
older class in methods/mindset.py remains only for historical replay/profiling,
not the active protocol. Use `PYTHONPATH=src` for direct library imports when
using the ordinary-pip repository installation.

## Study

The active protocol evaluates LoCoMo categories 1–4 (1,540 questions from all ten
conversations) and all 500 LongMemEval-S questions. The original 1,986-question
LoCoMo source audit is retained; the 446 adversarial questions are excluded from
this declared primary slice, not silently treated as ordinary answer targets.

The six methods are MINDSET, **LightMem (ACL 2026)**, Zep (2025 paper reproduction),
Nemori, Vector RAG and Long Context. LightMem's existing implementation is
unchanged. It is not the similarly named ICLR method.

LongMemEval contains 13 histories with repeated native session IDs. The adapter
preserves all 246,750 raw message occurrences, adds deterministic suffixes only
to colliding episode IDs, and keeps gold/retrieval session IDs unchanged.
The new occurrence-safe manifest and session-integrity audit are published;
the previous manifest and failed diagnostic are retained, not overwritten.

| Experiment | Generators | Methods |
|---|---|---|
| Primary | Qwen3.5-9B | All six |
| Family transfer | Llama-3.1-8B-Instruct | MINDSET, LightMem, Zep, Vector RAG |
| Scale | Qwen3.5-4B, Llama-3.2-3B-Instruct | Same four-method panel |
| Controller comparison | Qwen3.5-9B | Energy, rules, independent logistic classifier, nearest merge |

The transfer panel is prespecified, not an assertion that these are the strongest
measured baselines. Generative memory construction uses fixed Qwen3.5-9B.
Freeze evidence once per method/dataset/controller and reuse it across answer
models. Gemini is judge-only. No hosted generation fallback is used.

LoCoMo's primary conventional metric is its original category-specific native
F1. LongMemEval uses the official answer-checking prompt, with the judge-model
substitution disclosed. Supplementary LoCoMo semantic judging uses an explicitly
identified external published prompt; it is **not** the ACL LightMem evaluation
protocol or an identical literature/leaderboard setup.

## Installation

Use ordinary pip, Python 3.10 or newer, and an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv/Scripts/Activate.ps1
pip install -r requirements.txt
python -m pytest -q
python scripts/acl.py --help
```

Install `requirements-gpu.txt` separately on Linux for local checkpoint serving.
Do not use another project's environment or interfere with its allocated jobs.
DGX CPU preparation uses `hpc/dgx_cpu.slurm` with no GPU request.

## Development and validation

Independent synthetic development has 144 histories / 432 transitions, split
48/48/48 by history into training, validation and holdout. Exact IDs, labels and
hashes are published in `data/development/mindset_v2.json`.
A 3-by-3 validation search selected complexity 0.10 and switching cost 0.20.
This is calibration, not training of the encoder or test-set hand tuning.
Canonical synthetic success does not establish natural-dialogue generalization;
rules and the learned comparator also solve these easy holdout probes.

```bash
python scripts/acl.py calibrate --output outputs/acl/NEW_CALIBRATION
python scripts/acl.py validate-mindset --output outputs/acl/NEW_DIAGNOSTIC
python scripts/acl.py profile-controller --output outputs/acl/NEW_PROFILE --repetitions 3
```

Use fresh directories for timing and calibration outputs. The diagnostic samples
all LoCoMo conversation clusters and LongMemEval categories using score-independent
IDs, logs native-unit retrieval metrics, and makes no answer/judge/GPU calls.
It is not the full paper evaluation. Zep needs its local construction model;
Long Context has no native top-k ranking.

The active config remains **development_not_paper_frozen**. Final answer
generation is intentionally blocked until the validated protocol is frozen.
Old runs and the completed DGX v1 preparation are preserved and are not results
of the current controller.

## Bounded development before full runs

All experiment computation now runs on DGX through Slurm, including CPU
retrieval, calibration and tests. Do not run experiments on the laptop or DGX
login node. See [execution policy](AGENTS.md) and [HPC instructions](hpc/README.md).

Keep full paper runs blocked while MINDSET remains a development candidate.
The resumable CPU-only experiment uses a frozen 100-question cohort: 50 LoCoMo
questions across all ten conversations and 50 LongMemEval questions across all
six categories. Transition parameters stay at their independently calibrated
values; retrieval arms do not fit another feature-weight grid on these questions.
This is known-benchmark development, not an untouched confirmatory evaluation.

```bash
sbatch --nodelist=dgxa hpc/dgx_iteration.slurm
```

See [development protocol and novelty limitations](docs/development_iteration.md).
Outputs are under `outputs/acl/mindset_v2_development100_r01/`; repeated runs
resume exact completed cells. No full-run promotion, API calls or GPU jobs occur
in this stage. The ACL-2026 LightMem implementation is unchanged.

The diagnosis-driven [joint-energy prototype](docs/joint_energy_development.md)
has a separate calibration and job-specific outputs. Its DGX-only bounded test
uses `hpc/dgx_joint_iteration.slurm`; actual-answer end-to-end latency uses the
separate full-A100 `hpc/dgx_joint_answers.slurm`. LightMem is now requested at
eight hits, with its source unchanged and unequal internal candidate budgets
disclosed. All four requested latency columns are logged; CPU-only runs leave
end-to-end P50/P95 unmeasured. Neither prototype authorizes full paper runs.

## Paper pipeline

```bash
python scripts/acl.py freeze
python scripts/acl.py plan --output experiments/NEW_PLAN.json
python scripts/acl.py prepare --dataset locomo --method mindset --output outputs/acl/memory/locomo/mindset
# After explicitly freezing the final protocol and starting the local model:
python scripts/acl.py answer --prepared outputs/acl/memory/locomo/mindset --model qwen35_9b --output outputs/acl/runs/locomo/mindset/qwen35_9b
python scripts/acl.py judge --run outputs/acl/runs/locomo/mindset/qwen35_9b
python scripts/acl.py summarize --runs RUN1 RUN2 --output outputs/acl/reports
python scripts/acl.py statistics --runs RUN1 RUN2 --output outputs/acl/statistics
python scripts/acl.py efficiency --dataset locomo --output outputs/acl/NEW_EFFICIENCY --repetitions 3
```

All six methods require the same source histories and common evidence/context
budgets. Checkpoint commits, dataset/manifests, configuration, prompts, source,
calibration and generated evidence are hashed. Resume rejects mismatches.
Full efficiency uses fresh subprocesses, no score-cache reuse, repeated identical
hardware, and separate construction/answer/judge usage.

Read [protocol](docs/acl_protocol.md), [baseline fidelity](docs/baseline_fidelity.md),
[tuning guide](docs/mindset.md) and [remaining readiness work](docs/readiness.md).
No novelty, SOTA, QA gain or runtime improvement follows merely from this setup.

## Layout

```text
configs/             Protocol, checkpoint locks and judge configuration
data/manifests/      Exact benchmark IDs and source hashes
data/development/    Independent development histories and transition labels
src/mindset/         Memory, retrieval and evaluation implementation
src/mindset/acl/     Reproducible preparation, calibration, analysis and validation
scripts/acl.py       Experiment entry point
experiments/mindset/ Calibrated parameters and independent learned comparator
hpc/                 CPU and future GPU/Slurm scripts
tests/               Unit, mathematical, leakage and pipeline tests
legacy/              Preserved earlier workflow, excluded from publication
outputs/acl/         Generated outputs, excluded from Git
```

Private credentials, local caches, manuscripts and historical outputs are not
publication assets. Prepare an anonymous release separately; nothing is deleted
by this workflow.
