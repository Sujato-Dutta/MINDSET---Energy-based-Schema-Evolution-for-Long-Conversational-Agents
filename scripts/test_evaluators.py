from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindset.evaluation.locomo import evaluate_locomo
from mindset.evaluation.metrics import exact_match, token_f1


def main() -> None:
    dummy = [
        {
            "id": "multi",
            "prediction": "painting, hiking",
            "answer": "hiking, painting",
            "category": 1,
        },
        {
            "id": "temporal",
            "prediction": "7 May 2023",
            "answer": "7 May 2023",
            "category": 2,
        },
        {
            "id": "adversarial",
            "prediction": "No information available.",
            "answer": "",
            "category": 5,
        },
    ]
    report = {
        "generic_exact_match": exact_match("The blue house", "blue house"),
        "generic_token_f1": token_f1("blue house", "the blue house"),
        "locomo": evaluate_locomo(dummy),
    }
    print(json.dumps(report, indent=2))
    assert report["generic_exact_match"] == 1.0
    assert report["generic_token_f1"] == 1.0
    assert report["locomo"]["native_f1"] == 1.0


if __name__ == "__main__":
    main()
