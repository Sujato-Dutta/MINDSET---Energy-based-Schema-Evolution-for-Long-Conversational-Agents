"""Opt-in CPU scorer experiment, not a change to any study baseline."""
from __future__ import annotations


def cpu_int8_reranker(reranker):
    """Copy and dynamically quantize Linear layers; preserve original pipeline.

The installed CrossEncoder is an nn.Sequential pipeline, not a writable
`.model` property. Quantize the full copied pipeline so its pooling/head,
activation, tokenizer, prompts and predict interface remain intact. Numerical
equivalence is NOT assumed; quality and latency must be measured separately.
"""
    import torch
    if not isinstance(reranker, torch.nn.Module): raise TypeError('Torch scorer pipeline required')
    if any(p.device.type != 'cpu' for p in reranker.parameters()): raise ValueError('CPU-only conversion required')
    if not any(isinstance(m, torch.nn.Linear) for m in reranker.modules()): raise ValueError('No supported Linear layers')
    quantized = torch.ao.quantization.quantize_dynamic(reranker, {torch.nn.Linear}, dtype=torch.qint8, inplace=False)
    quantized.eval()
    if not any(isinstance(m, torch.ao.nn.quantized.dynamic.Linear) for m in quantized.modules()):
        raise RuntimeError('No dynamic INT8 layers were actually converted')
    return quantized


def tensor_bytes(model):
    """Logical parameter/buffer/unpacked-quantized-weight bytes, NOT serialization.

This excludes pickle/config objects and runtime packing/allocator overhead.
It is not peak RSS, complete pipeline storage or a restorable checkpoint size.
"""
    import torch
    weights = list(model.parameters()) + list(model.buffers())
    for module in model.modules():
        if isinstance(module, torch.ao.nn.quantized.dynamic.Linear):
            weights.append(module.weight())
            if module.bias() is not None: weights.append(module.bias())
    return sum(value.numel() * value.element_size() for value in weights)
