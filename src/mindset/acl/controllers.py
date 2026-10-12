from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import numpy as np
from mindset.memory.energy import TransitionSignals
from mindset.acl.io import read_json, write_json, digest_file

FEATURES = tuple(TransitionSignals.__dataclass_fields__)
ACTIONS = ("reinforce", "supersede", "split", "new")

def rule_controller(signals, config):
    if hasattr(config,"complexity_penalty"):
        if signals.contextual_potential: return "split",False
        if signals.temporal_potential: return "supersede",False
        return ("new" if (1-signals.dense_similarity)/2>config.complexity_penalty else "reinforce"),False
    if signals.compatibility < config.new_schema_threshold and not signals.topic_match:
        return "new", False
    if signals.contextual_potential > signals.temporal_potential + 0.12:
        return "split", False
    if signals.contradiction_probability >= 0.60 and (signals.explicit_change or signals.recurrence >= 0.5):
        return "supersede", False
    return "reinforce", False

def nearest_controller(signals, config):
    # One threshold; neither energies nor temporal/contextual transition decisions.
    if hasattr(config,"complexity_penalty"):
        return ("reinforce" if (1-signals.dense_similarity)/2<=config.complexity_penalty else "new"),False
    return ("reinforce" if signals.compatibility >= config.new_schema_threshold else "new"), False

class LearnedController:
    def __init__(self, path: str | Path):
        self.artifact = read_json(path)
        if self.artifact["features"] != list(FEATURES):
            raise ValueError("Classifier feature order mismatch")
        self.weights = np.asarray(self.artifact["coef"])
        self.bias = np.asarray(self.artifact["intercept"])
        self.classes = self.artifact["classes"]
    def choose(self,signals,config,allowed):
        x=np.asarray([getattr(signals,k) for k in FEATURES])
        logits=self.weights@x+self.bias
        valid=[i for i,c in enumerate(self.classes) if c in allowed]
        return self.classes[max(valid,key=lambda i:(float(logits[i]),-i))]
    def __call__(self, signals, config):
        x = np.asarray([getattr(signals, k) for k in FEATURES])
        logits = self.weights @ x + self.bias
        return self.classes[int(np.argmax(logits))], False

def fit_controller(training_path: str | Path, output: str | Path, forbidden_ids: set[str],
                   forbidden_histories: set[str] | None = None) -> None:
    from sklearn.linear_model import LogisticRegression
    data = read_json(training_path)
    # Actual independent action labels required. Never train on test histories or energy decisions.
    if data.get("label_source") not in {"human", "synthetic_ground_truth"}:
        raise ValueError("Training requires independently assigned human or synthetic-ground-truth labels")
    rows = data["records"]
    if forbidden_histories is not None:
        if any(not r.get("source_history_sha256") or r.get("source_history_sha256") in forbidden_histories
               or r.get("source_cluster_id") in forbidden_ids for r in rows):
            raise ValueError("Training history provenance missing or overlaps evaluation")
    if not rows or any(r["id"] in forbidden_ids or r.get("source_question_id") in forbidden_ids for r in rows):
        raise ValueError("Training split overlaps evaluation, or is empty")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate training IDs")
    if set(r["action"] for r in rows) != set(ACTIONS):
        raise ValueError("Independent training labels must include all four actions")
    x = np.asarray([[r["features"][f] for f in FEATURES] for r in rows])
    y = [r["action"] for r in rows]
    model = LogisticRegression(random_state=0, max_iter=1000).fit(x, y)
    write_json(output, {"features": list(FEATURES), "classes": model.classes_.tolist(),
                        "coef": model.coef_.tolist(), "intercept": model.intercept_.tolist(),
                        "training_sha256": digest_file(training_path), "label_source": data["label_source"],
                        "training_ids": [r["id"] for r in rows],
                        "source_cluster_ids":sorted({r["source_cluster_id"] for r in rows if r.get("source_cluster_id")}),
                        "source_history_sha256":sorted({r["source_history_sha256"] for r in rows if r.get("source_history_sha256")})}, immutable=True)

