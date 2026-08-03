# MINDSET: Energy-based Schema Evolution for Long Conversational Agents

MINDSET is a training-free memory controller for long-running conversational agents. It keeps dialogue as immutable, provenance-bearing episodes and organizes those episodes into versioned schemas through four explicit transitions: **reinforce**, **supersede**, **split**, and **new schema**. Transition selection minimizes a deterministic energy over representation distortion, contradiction, historical damage, fragmentation, and internal inconsistency. Memory construction and retrieval run locally; a frozen language model is used only to answer from the selected evidence.

## Final paper protocol

The final study contains **850 questions across two datasets**:

- **LoCoMo-700:** the frozen 700-question manifest at `data/manifests/locomo_primary700.json`.
- **MemoryAgentBench-150 (MAB):** the frozen 150-instance manifest at `data/manifests/memoryagentbench_final150.json`, comprising 125 in-context-learning and 25 recommendation instances.

The six final methods are:

1. **MINDSET (Ours)**
2. **Vector RAG**
3. **Long Context Memory**
4. **Nemori**
5. **MemoryBank**
6. **LightMem**

### Answer backbones

- **Primary model:** Cerebras `gpt-oss-120b`, used for all six methods on LoCoMo-700 and MAB-150.
- **Cross-backbone models:** Cerebras `zai-glm-4.7` and `gemma-4-31b`, each evaluated with all six methods on the complete LoCoMo-700 manifest.

No answer backbone is fine-tuned. Generation is deterministic (`temperature=0`), and GLM reasoning is disabled for the cross-backbone protocol.

All retrieval-based methods use the same bounded-evidence adapter: at most 20 candidate memories, 768-token passages with 96-token overlap, at most eight selected passages, an 8,000-token evidence cap, and at most three passages per source memory. Long Context is the structural exception because chronological history is its defining input; it truncates deterministically at 20,000 UTF-8 bytes.

## Headline results

- With GPT-OSS-120B on LoCoMo-700, MINDSET achieves **0.4280 Answer F1**, **0.6280 Recall@8**, **0.5572 MRR**, and **0.5453 nDCG@8**.
- On MAB-150, MINDSET leads the completed methods with **0.0240 native ICL exact match** and **0.1400 native recommendation Recall@5**.
- On the full LoCoMo-700 cross-backbone study, MINDSET ranks first with GLM-4.7 at **0.4080 Answer F1**. With Gemma-4-31B it scores **0.3306**, only 0.0002 behind LightMem, while retaining the best Recall@8, Precision@8, MRR, and nDCG@8.

## Implementation

MINDSET uses:

- `sentence-transformers/all-MiniLM-L6-v2` for local episode and passage embeddings;
- `cross-encoder/ms-marco-MiniLM-L6-v2` for local reranking;
- immutable raw episodes with timestamps and source identifiers;
- active and historical schemas with medoids, centroids, validity bounds, exceptions, and predecessor/successor links;
- direct dense/BM25 retrieval, schema-mediated provenance expansion, and version-aware routing.

No provider call is made during MINDSET ingestion or retrieval. The numerical transition settings are frozen engineering hyperparameters selected by hand tuning on a held-out development set with no overlap with the paper test manifests. The implementation in `src/mindset/memory/energy.py` and `src/mindset/methods/mindset.py` is the executable source of truth.

## Repository layout

```text
src/mindset/        Core memory methods, providers, datasets, and evaluation
scripts/            Experiment, scoring, audit, and analysis entry points
configs/            Provider/model and method configuration
data/manifests/     Frozen paper sample IDs and ordering
tests/              Unit and protocol tests
outputs/            Generated run artifacts (ignored except frozen run plans)
```

## Setup

Python 3.11 is the frozen environment used for the reported experiments.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Populate `CEREBRAS_API_KEYS` in `.env` with one or more comma-separated keys. Numbered variants such as `CEREBRAS_API_KEY_1` are also supported. The final paper generation runs require Cerebras credentials; Groq and Gemini variables remain optional for non-paper or legacy utilities. Credentials are read only from the environment and are not written to run artifacts.

## Data

Raw benchmark files are intentionally excluded from Git because of size and upstream licensing. Place them at:

```text
data/locomo/locomo10.json
data/memory_agent_bench/Test_Time_Learning-00000-of-00001.parquet
data/memory_agent_bench/entity2id.json
```

The exact paper sample IDs and order are versioned in `data/manifests/`. Dataset paths can also be supplied through the environment variables documented in `.env.example`.

## Validation and reproduction

Run the local test suite before launching provider-backed experiments:

```powershell
.\.venv\Scripts\python.exe scripts\test_evaluators.py
.\.venv\Scripts\python.exe -m pytest -q
```

Example primary LoCoMo execution:

```powershell
.\.venv\Scripts\python.exe scripts\run_locomo_experiment.py `
  --method mindset `
  --model gpt_oss_120b `
  --manifest data/manifests/locomo_primary700.json
```

Run the six frozen LoCoMo-700 MINDSET ablations with:

```powershell
.\.venv\Scripts\python.exe scripts\run_locomo_mindset_ablations.py `
  --plan outputs/plans/locomo_mindset_ablations700_v1.json `
  --state outputs/orchestration/reproduction_ablations700.json
```

Run all twelve full LoCoMo-700 GLM/Gemma cells with:

```powershell
.\.venv\Scripts\python.exe scripts\run_locomo_cross_backbones.py `
  --plan outputs/plans/locomo_cross_backbone700_glm_gemma_v1.json `
  --state outputs/orchestration/reproduction_cross700.json
```

These commands make paid provider calls. Run configurations, predictions, metrics, telemetry, and snapshots are written beneath `outputs/`.

## Baseline fidelity

- **MemoryBank** follows its dialogue memory, summary, personality, and forgetting architecture, using the shared GPT-OSS backbone for generated summaries.
- **LightMem** is a paper-guided CPU-local reproduction. Unreleased selector/consolidation artifacts are replaced by the pinned pretrained cross-encoder and deterministic medoid-graph consolidation.
- **Nemori** is a pinned CPU-local reproduction of semantic boundary alignment and predict-calibrate memory, with documented extractive/local substitutions.

Results apply to the repository implementations described in the paper; they should not be presented as unqualified reruns of unavailable upstream training artifacts.
