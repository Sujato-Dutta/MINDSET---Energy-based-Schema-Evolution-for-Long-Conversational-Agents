"""Isolated r03 warning: noun-tagged opaque heads stay uncertain too."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("preserved_binding_candidate", Path(__file__).with_name("source_binding_candidate.py"))
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)


def reference_uncertainty(binding, reports, docs):
    result = base.reference_uncertainty(binding, reports, docs)
    if binding["kind"] != "attribute_change":
        return result
    warnings = list(result["warnings"])
    for quote in binding["bindings"]:
        sid = quote["source_id"]
        for token in docs[sid]:
            if not any(a <= token.idx and token.idx+len(token.text) <= b for a,b in quote["occurrences"]):
                continue
            if token.dep_ in base.ARGUMENTS and token.pos_ in ("NOUN", "PROPN") and base.base.lemma(token) in {"one", "someone", "something", "anyone", "anything"}:
                warnings.append({"source_id": sid, "span": [token.idx, token.idx+len(token.text)], "text": token.text,
                    "reason": "opaque_argument_head_independent_of_pronoun_tag"})
    return {**result, "warnings": warnings,
            "update_proposal": binding["provisional_update_proposal"] and not warnings}
