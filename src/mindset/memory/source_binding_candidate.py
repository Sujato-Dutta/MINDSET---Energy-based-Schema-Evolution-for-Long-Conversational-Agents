"""r02 source-binding candidate; preserves the failed r01 module and worker.

Normalization is unchanged. A separate conservative syntax warning recognizes
unresolved possessive holders as well as subject/object references. Not a novel
coreference method and not a semantic truth certificate.
"""
from pathlib import Path
import importlib.util
from mindset.utils.hashing import stable_hash

spec = importlib.util.spec_from_file_location("preserved_source_binding", Path(__file__).with_name("source_binding.py"))
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
normalize_binding = base.normalize_binding
OPAQUE = base.OPAQUE | {"his", "hers", "their", "theirs", "its", "you", "your", "yours", "we", "us", "our", "ours"}
ARGUMENTS = base.ARGUMENTS | {"poss"}


def reference_uncertainty(binding, reports, docs):
    sources = base.reports_by_id(reports)
    if set(docs) != {"A", "B"} or any(docs[s].text != sources[s]["text"] for s in docs):
        raise ValueError("Parser documents differ from complete source text")
    if binding["source_reports_sha256"] != stable_hash(reports):
        raise ValueError("Binding provenance differs")
    issues = []; resolved = []
    if binding["kind"] != "attribute_change":
        return {"warnings": [], "resolved_nominal_links": [], "update_proposal": False,
                "scope": "non-change never mutates state"}
    for quote in binding["bindings"]:
        sid = quote["source_id"]; other = "B" if sid == "A" else "A"
        covered = [t for t in docs[sid] if any(a <= t.idx and t.idx+len(t.text) <= b for a,b in quote["occurrences"])]
        for token in covered:
            if token.dep_ not in ARGUMENTS:
                continue
            if (token.pos_ == "PRON" or token.dep_ == "poss") and base.lemma(token) in OPAQUE:
                issues.append({"source_id": sid, "span": [token.idx, token.idx+len(token.text)],
                    "text": token.text, "reason": "unresolved_pronominal_or_possessive_argument"})
                continue
            if token.pos_ not in ("NOUN", "PROPN") or not base.deictic_children(token):
                continue
            candidates = []
            if sources[sid]["speaker"] == sources[other]["speaker"]:
                candidates = [t for t in docs[other] if base.lemma(t) == base.lemma(token)
                    and t.pos_ in ("NOUN", "PROPN") and not base.deictic_children(t)
                    and any(c.dep_ == "poss" and c.text.casefold() == "my" for c in t.children)]
            if len(candidates) == 1:
                antecedent = candidates[0]
                resolved.append({"source_id": sid, "reference_span": [token.idx, token.idx+len(token.text)],
                    "antecedent_source_id": other, "antecedent_span": [antecedent.idx, antecedent.idx+len(antecedent.text)],
                    "reason": "unique_same_lemma_explicit_self_possession_not_semantic_identity_proof"})
            else:
                issues.append({"source_id": sid, "span": [token.idx, token.idx+len(token.text)],
                    "text": token.text, "reason": "unresolved_deictic_nominal", "candidate_count": len(candidates)})
    return {"warnings": issues, "resolved_nominal_links": resolved,
            "update_proposal": binding["provisional_update_proposal"] and not issues,
            "scope": "syntax-based conservative warning; not verified identity/chronology/world truth"}
