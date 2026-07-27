from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["correct", "partially_correct", "incorrect", "unsupported"]
    rationale: str = Field(max_length=1000)


def build_blind_judge_prompt(
    question: str, reference: str, prediction: str, evidence: str | None = None
) -> str:
    sections = [
        "Judge the candidate answer without guessing which system produced it.",
        f"QUESTION:\n{question}",
        f"REFERENCE ANSWER:\n{reference}",
        f"CANDIDATE ANSWER:\n{prediction}",
    ]
    if evidence is not None:
        sections.append(f"OPTIONAL EVIDENCE:\n{evidence}")
    sections.append(
        "Return strict JSON with verdict in {correct, partially_correct, incorrect, unsupported} "
        "and a concise rationale. Mark unsupported when the answer asserts facts absent from evidence."
    )
    return "\n\n".join(sections)
