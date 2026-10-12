"""Latency definitions shared by development and later frozen paper reports."""
import numpy as np

def latency_summary(rows):
    """Milliseconds, linear quantiles, with no imputed end-to-end timings.

    Retrieval includes the native retrieval and common raw-evidence packet.
    End-to-end starts before retrieval and stops after the real answer returns;
    ingestion, server startup, judging and cache-hit requests are excluded.
    """
    rows=list(rows)
    def quantiles(field):
        vals=[r[field] for r in rows if r.get(field) is not None and not r.get("cache_hit",False)]
        if any(not np.isfinite(x) or x<0 for x in vals): raise ValueError("Invalid latency")
        if not vals: return None,None,0
        p50,p95=np.quantile(vals,[.5,.95],method="linear")*1000
        return float(p50),float(p95),len(vals)
    r50,r95,rn=quantiles("retrieval_seconds")
    e50,e95,en=quantiles("end_to_end_seconds")
    if en and en!=rn: raise ValueError("End-to-end panel must be complete and paired with retrieval")
    return {"retrieval_latency_p50_ms":r50,"retrieval_latency_p95_ms":r95,
            "end_to_end_p50_ms":e50,"end_to_end_p95_ms":e95,
            "retrieval_queries":rn,"end_to_end_queries":en,
            "end_to_end_status":"measured_actual_answers" if en else "not_measured_no_answer_calls"}
