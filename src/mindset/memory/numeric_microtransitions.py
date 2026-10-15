"""Explicit from/to numeric report candidates inside immutable messages.

Narrow English syntax coverage, not comprehensive state extraction or world truth.
Source values/qualification spans are retained; factual dates are never invented.
"""
import importlib.util
from pathlib import Path
from mindset.utils.hashing import stable_hash

spec = importlib.util.spec_from_file_location("preserved_numeric_projection", Path(__file__).with_name("report_frames.py"))
numeric = importlib.util.module_from_spec(spec); spec.loader.exec_module(numeric)
CHANGE = {"change", "increase", "decrease", "jump", "rise", "fall", "grow", "drop", "reduce", "raise", "shift"}
NONASSERTIVE = {"say", "tell", "ask", "quote", "hear", "read", "think", "believe", "hope", "wish", "plan"}
MODALS = {"would", "could", "might", "may", "should", "will", "can"}


def span(tokens, text):
    tokens = sorted(tokens, key=lambda t: t.i)
    if not tokens:
        return None
    a = tokens[0].idx; b = tokens[-1].idx + len(tokens[-1].text)
    return {"char_start": a, "char_end": b, "text": text[a:b]}


def value(prep, text):
    objects = [c for c in prep.children if c.dep_ == "pobj"]
    if len(objects) != 1:
        return None
    obj = objects[0]
    if obj.pos_ == "NUM":
        head = obj
    else:
        nums = [c for c in obj.children if c.dep_ == "nummod"]
        if len(nums) != 1:
            return None
        head = nums[0]
    # Some parses attach the subsequent 'to' phrase below the first numeric
    # object. It is not part of that endpoint's literal value. Never bridge a
    # removed interior phrase into a new, invented source interval.
    excluded = {t.i for child in head.children if child.dep_ == "prep"
                for t in child.subtree}
    tokens = sorted((t for t in head.subtree if t.i not in excluded), key=lambda t:t.i)
    if tokens and any(tokens[0].i <= i <= tokens[-1].i for i in excluded):
        return None
    if any(t.ent_type_ in ("DATE", "TIME") or t.dep_ == "conj" for t in tokens):
        return None
    literal = span(tokens, text)
    number = numeric.normalize_number(literal["text"]) if literal else None
    return {**literal, "numeric_normalization_alternative": number} if number is not None else None


def collapsed_from_to(prep, doc, text):
    """spaCy may parse 'from 900 to 700' with both modifiers below 700.

    Require the explicit literal delimiter and a single numeric object; do not
    infer a missing endpoint or normalize away approximation/qualification.
    """
    objects = [c for c in prep.children if c.dep_ == "pobj" and c.pos_ == "NUM"]
    if len(objects) != 1:
        return None
    obj = objects[0]
    separators = [c for c in obj.children if c.dep_ == "quantmod" and c.pos_ == "ADP" and numeric.lemma(c) == "to"]
    if len(separators) != 1:
        return None
    separator = separators[0]
    if not prep.i < separator.i < obj.i:
        return None
    sides = [list(doc[prep.i+1:separator.i]), list(doc[separator.i+1:obj.i+1])]
    endpoints = []
    for tokens in sides:
        if not tokens or any(t.ent_type_ in ("DATE", "TIME") for t in tokens):
            return None
        literal = span(tokens, text)
        number = numeric.normalize_number(literal["text"])
        if number is None:
            return None
        endpoints.append({**literal, "numeric_normalization_alternative": number})
    return endpoints


def compile_numeric_reports(doc, text, speaker, *, episode_id, observation_time):
    if doc.text != text:
        raise ValueError("Parser/source text differs")
    if "```" in text or text.lstrip().startswith(("Web search results:", "URL:")):
        return []
    result = []
    for sent in doc.sents:
        if sent.text.rstrip().endswith("?") or any(t.is_quote for t in sent):
            continue
        for root in sent:
            if root.pos_ != "VERB" or numeric.lemma(root) not in CHANGE:
                continue
            chain = [root] + list(root.ancestors)
            if any(numeric.lemma(t) in NONASSERTIVE for t in chain):
                continue
            if any(c.dep_ == "neg" or (c.dep_ in ("aux", "auxpass") and numeric.lemma(c) in MODALS)
                   or (c.dep_ == "mark" and numeric.lemma(c) in ("if", "unless")) for t in chain for c in t.children):
                continue
            froms = [t for t in root.subtree if numeric.lemma(t) == "from" and t.dep_ == "prep" and numeric.predicate(t.head) == root]
            tos = [t for t in root.subtree if numeric.lemma(t) == "to" and t.dep_ == "prep" and numeric.predicate(t.head) == root]
            if len(froms) != 1:
                continue
            if len(tos) == 1 and froms[0].i < tos[0].i:
                before = value(froms[0], text); after = value(tos[0], text)
            elif not tos:
                pair = collapsed_from_to(froms[0], doc, text)
                if pair is None:
                    continue
                before, after = pair
            else:
                continue
            if not before or not after or before["numeric_normalization_alternative"] == after["numeric_normalization_alternative"]:
                continue
            own = numeric.subjects(root)
            if len(own) != 1:
                continue
            subject = own[0]
            # No guessed 'your', 'their' or unknown holder. Owned subject is an
            # explicitly named scope in the report, not proof it is a scalar.
            if any(c.dep_ == "poss" and c.text.casefold() == "my" for c in subject.children):
                property_head = subject
            elif subject.text.casefold() == "i":
                owned = [c for c in root.children if c.dep_ == "dobj" and
                         any(p.dep_ == "poss" and p.text.casefold() == "my" for p in c.children)]
                if len(owned) != 1:
                    continue
                property_head = owned[0]
            else:
                continue
            property_tokens = [property_head] + [c for c in property_head.children if c.dep_ in ("compound", "amod", "poss", "det")]
            property_span = span(property_tokens, text)
            qualifiers = [{"label": e.label_, "char_start": e.start_char, "char_end": e.end_char, "text": e.text}
                          for e in sent.ents if e.label_ in ("DATE", "TIME") and not
                          (before["char_start"] <= e.start_char < before["char_end"] or after["char_start"] <= e.start_char < after["char_end"])]
            source = {"episode_id": episode_id, "speaker": speaker, "observation_time": observation_time,
                      "source_sha256": stable_hash(text)}
            record = {"source": source, "property_span": property_span,
                "predicate_span": span([root], text), "before_value": before, "after_value": after,
                "scope_span": {"char_start": sent.start_char, "char_end": sent.end_char, "text": sent.text},
                "factual_time_qualifiers": qualifiers, "ordering": "source_explicit_from_to",
                "owner_binding": "explicit_self_possession_in_report", "candidate_not_certified_scalar_attribute": True,
                "factual_interval_normalized": False, "semantic_correctness_verified": False, "world_truth_verified": False}
            record["id"] = stable_hash(record)
            result.append(record)
    return result
