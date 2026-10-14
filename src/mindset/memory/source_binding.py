"""Source-only provenance normalization and conservative reference uncertainty.

Neither unique literal binding nor a syntax check proves semantic identity or
world truth. This repair never changes relationship labels/factual direction.
"""
from itertools import permutations
from mindset.utils.hashing import stable_hash

KINDS = ("attribute_change", "event_progression", "insufficient")
DIRECTIONS = ("a_to_b", "b_to_a", "none", "unknown")
FIELDS = {"kind", "variable", "a_quote", "b_quote", "direction"}
ARGUMENTS = {"nsubj", "nsubjpass", "dobj", "pobj", "attr", "oprd", "dative"}
OPAQUE = {"it", "he", "she", "they", "them", "him", "her", "this", "that", "these", "those",
          "one", "another", "someone", "something", "anyone", "anything"}
DEICTIC = {"this", "that", "these", "those", "other", "another"}


def reports_by_id(reports):
    if not isinstance(reports, list) or len(reports) != 2 or {r.get("source_id") for r in reports} != {"A", "B"}:
        raise ValueError("Two unique complete source reports required")
    for row in reports:
        if set(row) != {"source_id", "speaker", "observation_time", "text"} or not all(isinstance(v, str) for v in row.values()):
            raise ValueError("Invalid source report fields")
    return {r["source_id"]: r for r in reports}


def offsets(text, quote):
    """Retain every literal occurrence; never invent or silently choose an offset."""
    if not quote:
        return []
    result = []; start = 0
    while True:
        index = text.find(quote, start)
        if index < 0:
            return result
        result.append([index, index + len(quote)]); start = index + 1


def normalize_binding(payload, reports):
    sources = reports_by_id(reports)
    if not isinstance(payload, dict) or set(payload) != FIELDS or not all(isinstance(v, str) for v in payload.values()):
        raise ValueError("Invalid contrast shape")
    if payload["kind"] not in KINDS or payload["direction"] not in DIRECTIONS:
        raise ValueError("Unknown contrast label")
    positive = payload["kind"] == "attribute_change"
    if positive and (not payload["variable"].strip() or payload["direction"] == "none"):
        raise ValueError("Missing variable or chronology")
    if not positive and payload["direction"] != "none":
        raise ValueError("Non-change must not have temporal direction")
    quotes = [payload["a_quote"], payload["b_quote"]]
    if positive and (not all(q.strip() for q in quotes) or quotes[0] == quotes[1]):
        raise ValueError("Missing or identical positive witnesses")
    fits = [p for p in permutations(("A", "B"))
            if all(not q or q in sources[sid]["text"] for q, sid in zip(quotes, p))]
    if ("A", "B") in fits:
        assignment = ("A", "B"); status = "declared_source_binding"
    elif len(fits) == 1:
        assignment = fits[0]; status = "unique_literal_source_rebinding"
    else:
        raise ValueError("No unique literal source assignment")
    bindings = [{"source_id": sid, "quote": quote,
                 "occurrences": offsets(sources[sid]["text"], quote),
                 "original_field": field}
                for quote, sid, field in zip(quotes, assignment, ("a_quote", "b_quote")) if quote]
    return {"kind": payload["kind"], "direction": payload["direction"],
        "variable": payload["variable"] if positive else "",
        "bindings": bindings, "binding_status": status,
        "nonchange_extra_fields_ignored_for_state": not positive and any(payload[k] for k in ("variable", "a_quote", "b_quote")),
        "source_reports_sha256": stable_hash(reports),
        "label_and_direction_unchanged": True,
        "provisional_update_proposal": positive and payload["direction"] != "unknown",
        "semantic_correctness_verified": False, "world_truth_or_update_verified": False}


def lemma(token):
    return (token.lemma_ or token.text).casefold()


def deictic_children(head):
    return [c for c in head.children if c.dep_ in ("det", "amod") and lemma(c) in DEICTIC]


def reference_uncertainty(binding, reports, docs):
    """A conservative syntax warning, not general coreference resolution.

    Bare third-person/deictic arguments stay unresolved. A deictic nominal may
    link only to a unique same-lemma, directly 'my'-possessed nominal from the
    other complete report by the same speaker. No name/holder is guessed.
    Missed warnings are NOT proof the remaining interpretation is correct.
    """
    sources = reports_by_id(reports)
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
        doc = docs[sid]
        # A repeated quote has multiple possible occurrences. Inspect their
        # union, preserving every offset; this is not an inferred alignment.
        covered = [t for t in doc if any(a <= t.idx and t.idx+len(t.text) <= b for a,b in quote["occurrences"])]
        for token in covered:
            if token.dep_ not in ARGUMENTS:
                continue
            if token.pos_ == "PRON" and lemma(token) in OPAQUE:
                issues.append({"source_id": sid, "span": [token.idx, token.idx+len(token.text)],
                    "text": token.text, "reason": "unresolved_pronominal_argument"})
                continue
            if token.pos_ not in ("NOUN", "PROPN") or not deictic_children(token):
                continue
            candidates = []
            if sources[sid]["speaker"] == sources[other]["speaker"]:
                candidates = [t for t in docs[other] if lemma(t) == lemma(token)
                    and t.pos_ in ("NOUN", "PROPN") and not deictic_children(t)
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
