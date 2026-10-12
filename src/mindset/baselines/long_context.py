"""Chronological full context under a shared total token budget."""
from __future__ import annotations
from mindset.acl.clients import messages,prompt_tokens

def render_selected(selected):
    return "\n\n".join(f"[{i}] {e['text']}" for i,e in enumerate(selected,1))

def fit_long_context(question,date,selected,tokenizer,context_tokens,output_tokens,thinking=None):
    selected=list(selected)
    def fits(start):
        subset=selected[start:]; text=render_selected(subset)
        prompt=messages(question,text,date)
        return prompt_tokens(tokenizer,prompt,thinking=thinking)+output_tokens<=context_tokens
    if not fits(len(selected)):
        raise ValueError("Question itself exceeds context budget")
    # Binary search the minimal oldest-event omission count, then verify actual template size.
    low,high=0,len(selected)
    while low<high:
        middle=(low+high)//2
        if fits(middle): high=middle
        else: low=middle+1
    kept=selected[low:]
    return kept,render_selected(kept),low
