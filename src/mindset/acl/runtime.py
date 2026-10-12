from __future__ import annotations
import threading
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import psutil
from mindset.acl.io import write_json
from mindset.baselines.event_rag import EventRAGIndex
from mindset.baselines.lightmem import LightMemACL2026Reproduction
from mindset.baselines.zep import ZepPaperReproduction
from mindset.methods.mindset import MindsetMemory, MindsetConfig
from mindset.methods.compact import CompactMindsetMemory, CompactConfig
from mindset.methods.nemori import NemoriFaithfulReproduction
from mindset.acl.controllers import rule_controller, nearest_controller, LearnedController

class CachedEncoder:
    """In-process reuse only; create a fresh instance for each timing repetition."""
    def __init__(self, encoder, enabled=True):
        self.encoder=encoder; self.enabled=enabled; self.cache={}
    def encode(self,texts,**kwargs):
        texts=list(texts)
        if not self.enabled: return self.encoder.encode(texts,**kwargs)
        # Experiment callers all request normalized numpy vectors.
        if kwargs.get("normalize_embeddings") is not True or kwargs.get("convert_to_numpy") is not True:
            return self.encoder.encode(texts,**kwargs)
        missing=list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            vectors=self.encoder.encode(missing,**kwargs)
            self.cache.update(zip(missing,vectors))
        return np.asarray([self.cache[t] for t in texts])
    def __getattr__(self,key): return getattr(self.encoder,key)

class ResourceMonitor:
    """10 ms RSS sampling and optional whole-device VRAM; scopes recorded in artifacts."""
    def __enter__(self):
        self.peak_rss=0; self.peak_vram=None; self.stop=threading.Event()
        self.process=psutil.Process()
        self.nvml=None
        try:
            import os
            if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
                raise RuntimeError("GPU monitoring disabled for explicit CPU-only allocation")
            import pynvml
            pynvml.nvmlInit(); self.nvml=pynvml
            visible=os.environ.get("CUDA_VISIBLE_DEVICES")
            self.gpus=[]
            if visible:
                for identifier in visible.split(","):
                    identifier=identifier.strip()
                    if identifier.startswith(("MIG-","GPU-")):
                        self.gpus.append(pynvml.nvmlDeviceGetHandleByUUID(identifier))
                    elif identifier.isdigit():
                        self.gpus.append(pynvml.nvmlDeviceGetHandleByIndex(int(identifier)))
            # Unknown allocation visibility means missing VRAM data, never a host-wide sweep.
        except Exception: self.gpus=[]
        def sample():
            while not self.stop.is_set():
                self.peak_rss=max(self.peak_rss,self.process.memory_info().rss)
                if self.gpus:
                    used=sum(self.nvml.nvmlDeviceGetMemoryInfo(h).used for h in self.gpus)
                    self.peak_vram=max(self.peak_vram or 0,used)
                self.stop.wait(0.01)
        self.thread=threading.Thread(target=sample,daemon=True); self.thread.start()
        return self
    def __exit__(self,*exc):
        self.stop.set(); self.thread.join()
    def report(self):
        return {"peak_process_rss_bytes":self.peak_rss,"peak_device_vram_bytes":self.peak_vram,
                "vram_scope":"explicitly_visible_allocated_devices_including_server; no_hostwide_fallback",
                "rss_scope":"memory_worker_process_excluding_external_server"}

def make_method(name,events,config,encoder,reranker,client=None,controller="energy",learned_path=None):
    if name=="vector_rag": return EventRAGIndex(events,encoder)
    if name=="long_context": return None
    if name=="lightmem": method=LightMemACL2026Reproduction(encoder,reranker)
    elif name=="nemori": method=NemoriFaithfulReproduction(encoder)
    elif name=="zep": method=ZepPaperReproduction(encoder,reranker,client)
    elif name=="mindset":
        policy={"energy":None,"rule":rule_controller,"nearest":nearest_controller}
        if controller=="learned":
            if not learned_path: raise ValueError("Frozen learned controller artifact required")
            selected=LearnedController(learned_path)
        else: selected=policy[controller]
        params=dict(config["mindset"]); profile=params.pop("profile","v1")
        klass,cfgklass=(CompactMindsetMemory,CompactConfig) if profile=="compact_v2" else (MindsetMemory,MindsetConfig)
        method=klass(cfgklass(**params),encoder=encoder,local_reranker=reranker,transition_controller=selected)
    else: raise ValueError(f"Unknown method: {name}")
    method.ingest(events)
    return method

def snapshot(method,events,path):
    if method is None:
        write_json(path,{"event_ids":[e.event_id for e in events],"raw_events_file":"raw_events.json"})
    elif hasattr(method,"snapshot"):
        method.snapshot(path)
        # Store index vectors as part of storage measurement, not just JSON metadata.
        arrays={k:v for k,v in vars(method).items() if isinstance(v,np.ndarray)}
        if arrays: np.savez_compressed(Path(path).with_suffix(".npz"),**arrays)
    else:
        write_json(path,{"event_ids":[e.event_id for e in events],"raw_events_file":"raw_events.json"})
        np.savez_compressed(Path(path).with_suffix(".npz"),vectors=method.vectors)

def distribution(values):
    v=np.asarray(values,dtype=float)
    if not len(v): return {"count":0,"mean":None,"median":None,"p95":None,"std":None,"total":0}
    return {"count":len(v),"mean":float(v.mean()),"median":float(np.median(v)),
            "p95":float(np.quantile(v,.95)),"std":float(v.std()),"total":float(v.sum())}

