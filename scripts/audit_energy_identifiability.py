"""Frozen original100 energy-decision necessity audit. DGX Slurm only.

Replays the original Joint ingestion, logs every feasible post-state decision,
and removes one price at a time without changing the current state or rules.
This tests decision influence, NOT beneficial influence, end-to-end ablations,
answer superiority, or literature novelty. No retrieval or answer calls.
"""
import os
import hashlib
import json
from dataclasses import asdict
from collections import Counter
from pathlib import Path
from time import perf_counter

from mindset.acl.pipeline import ROOT, record_environment, write_csv
from mindset.acl.joint_iteration import protocol, groups_for
from mindset.acl.runtime import CachedEncoder
from mindset.acl.io import read_json, write_json, unique_rows, append_row, digest_file
from mindset.methods.compact import CompactConfig
from mindset.methods.joint import JointMindsetMemory
from mindset.evaluation.energy_identifiability import attribute, TERMS


class ObservedJoint(JointMindsetMemory):
    """Observer only: all feasible plans, choices and state mutations unchanged."""
    def _plans(self, schema, episode):
        plans = super()._plans(schema, episode)
        self._observed_signals[schema.schema_id] = asdict(self._signals(schema, episode))
        return plans

    def _ingest_episode(self, episode):
        self._observed_signals = {}
        super()._ingest_episode(episode)
        certificate = self._decision_certificates.get(episode.episode_id)
        row = {"episode_id": episode.episode_id,
               "operation": self.operation_log._records[-1].operation,
               "certificate": certificate,
               "signals_by_target": self._observed_signals,
               "attribution": attribute(certificate) if certificate else None}
        self._observer_rows.append(row)


def run():
    if os.name == "nt" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("DGX Slurm only")
    manifest = ROOT / "data/manifests/mindset_v2_development100.json"
    calibration = ROOT / "outputs/acl/mindset_joint_v2_27058/calibration/calibration.json"
    cfg, cohort, frozen = protocol(manifest, calibration)
    output = ROOT / f"outputs/acl/energy_identifiability_{os.environ['SLURM_JOB_ID']}"
    output.mkdir(parents=True, exist_ok=True)
    sources = ["methods/joint.py", "methods/compact.py", "methods/mindset.py", "memory/energy.py",
               "evaluation/energy_identifiability.py"]
    write_json(output / "audit.json", {
        "manifest_sha256": digest_file(manifest), "calibration_sha256": digest_file(calibration),
        "worker_sha256": digest_file(__file__),
        "modules": {s: digest_file(ROOT / "src/mindset" / s) for s in sources},
        "checkpoint_lock_sha256": digest_file(ROOT / "configs/checkpoints.lock.json"),
        "parameters": frozen["selected_parameters"],
        "selection": "all histories underlying original100, no QA-outcome filtering",
        "scope": "one-step necessity, eligibility held fixed; not benefit/QA/novelty proof",
        "qa_calls": 0, "judge_calls": 0, "full_runs_authorized": False,
    }, immutable=True)
    record_environment(output)
    from sentence_transformers import SentenceTransformer
    locked = read_json(ROOT / "configs/checkpoints.lock.json")["encoder"]
    encoder = SentenceTransformer(locked["checkpoint"], revision=locked["revision"],
                                  device=cfg["retrieval"]["device"])
    encoder = CachedEncoder(encoder, enabled=True)
    done = unique_rows(output / "histories.jsonl")
    reports = []
    expected_histories = []
    for dataset, cluster, queries in groups_for(cohort, cfg):
        key = f"{dataset}/{cluster}"; expected_histories.append(key)
        if key in done:
            continue
        parameters = {**{k: v for k, v in cfg["mindset"].items() if k != "profile"},
                      **frozen["selected_parameters"]}
        memory = ObservedJoint(CompactConfig(**parameters), encoder=encoder, local_reranker=None)
        memory._observer_rows = []
        started = perf_counter(); memory.ingest(queries[0].events)
        invariants = memory.validate_invariants()
        history = {"id": key, "dataset": dataset, "cluster": cluster,
                   "question_ids": [q.id for q in queries],
                   "events_sha256": hashlib.sha256(
                       json.dumps([asdict(e) for e in queries[0].events],
                           sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                   "episodes": len(memory._observer_rows), "invariants": invariants,
                   "observed_replay_seconds": perf_counter() - started,
                   "not_a_paper_latency_measurement": True}
        write_json(output / "decisions" / f"{dataset}-{cluster}.json", memory._observer_rows, immutable=True)
        append_row(output / "histories.jsonl", history); done[key] = history
        encoder.cache.clear()
        print(f"Completed {key}: {history['episodes']} decisions", flush=True)
    if set(done) != set(expected_histories):
        raise ValueError("History cohort mismatch")
    for dataset in ("locomo", "longmemeval"):
        rows = []
        for history in done.values():
            if history["dataset"] == dataset:
                rows.extend(read_json(output / "decisions" / f"{dataset}-{history['cluster']}.json"))
        attributed = [r["attribution"] for r in rows if r["attribution"] is not None]
        counts = Counter(r["operation"] for r in rows)
        common = {"dataset": dataset, "episodes": len(rows), "initial_creations": len(rows)-len(attributed),
                  "evaluated_decisions": len(attributed),
                  "multiple_actions": sum(r["distinct_actions"] > 1 for r in attributed),
                  "multiple_targets": sum(r["distinct_targets"] > 1 for r in attributed),
                  "singleton_candidates": sum(r["candidate_count"] == 1 for r in attributed),
                  "operation_counts": dict(counts)}
        for term in (*TERMS, "energy_off"):
            reports.append({**common, "removed_term": term,
                **{field: sum(r["controls"][term][field] for r in attributed)
                   for field in ("decision_changed", "action_changed", "mutation_changed", "target_changed")},
                "scope": "same-state decision influence, not useful contribution or QA ablation"})
    write_csv(output / "component_influence.csv", reports)
    write_json(output / "analysis.json", {"reports": reports, "histories": len(done),
        "questions": len(cohort["records"]), "no_new_answers": True,
        "full_runs_authorized": False, "useful_energy_or_novelty_proven": False}, immutable=True)
    write_json(output / "status.json", {"status": "complete_mechanism_audit", "histories": len(done),
        "questions": len(cohort["records"]), "qa_calls": 0, "judge_calls": 0})
    print(reports, flush=True)


if __name__ == "__main__":
    run()
