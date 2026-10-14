"""Explicitly audited r02 adapter to the preserved source-binding replay worker."""
import os
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mindset.acl.io import write_json, digest_file


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    if os.name == "nt" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("DGX Slurm only")
    base_path = ROOT / "scripts/repair_source_contrasts.py"
    candidate_path = ROOT / "src/mindset/memory/source_binding_candidate.py"
    worker = load("preserved_repair_worker", base_path)
    candidate = load("source_binding_candidate", candidate_path)
    # Only the instrument's pure source-binding strategy is changed. The
    # caller's original labels, expected outputs and frozen input gate remain.
    worker.binding_module = candidate
    adapter = ROOT / f"outputs/acl/source_binding_adapter_{os.environ['SLURM_JOB_ID']}"
    write_json(adapter / "audit.json", {
        "launcher_sha256": digest_file(__file__), "preserved_worker_sha256": digest_file(base_path),
        "preserved_binding_sha256": digest_file(ROOT / "src/mindset/memory/source_binding.py"),
        "active_binding_candidate_sha256": digest_file(candidate_path),
        "replay_output": f"outputs/acl/source_binding_repair_{os.environ['SLURM_JOB_ID']}",
        "strategy": "same literal source normalization, r02 explicit possessive-reference warnings",
        "original_failed_attempt": "27687", "new_inference_calls": 0}, immutable=True)
    worker.run()
