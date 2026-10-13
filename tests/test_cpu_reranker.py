import numpy as np
import pytest
import torch
from mindset.acl.cpu_reranker import cpu_int8_reranker, tensor_bytes


class Scorer(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.linear = torch.nn.Linear(4, 1)

    def predict(self, pairs, **kwargs):
        with torch.inference_mode():
            inputs = torch.ones((len(pairs), 4))
            return self.linear(inputs).squeeze(-1).numpy()


def test_cpu_conversion_does_not_replace_original_layers_or_predict_interface():
    scorer = Scorer().eval(); before = scorer.linear.weight.detach().clone()
    copied = cpu_int8_reranker(scorer)
    assert isinstance(scorer.linear, torch.nn.Linear)
    torch.testing.assert_close(scorer.linear.weight, before)
    result = copied.predict([('q', 'e'), ('q2', 'e2')])
    assert result.shape == (2,) and np.isfinite(result).all()
    assert isinstance(copied.linear, torch.ao.nn.quantized.dynamic.Linear)
    assert tensor_bytes(copied) > 0 and tensor_bytes(scorer) > 0


def test_unsupported_pipeline_fails_explicitly():
    with pytest.raises(TypeError): cpu_int8_reranker(object())
    with pytest.raises(ValueError): cpu_int8_reranker(torch.nn.Identity())


def test_storage_measurement_does_not_pickle_arbitrary_pipeline_extra_state():
    class MetadataScorer(Scorer):
        def get_extra_state(self): return {'cannot_pickle': lambda value: value}
        def set_extra_state(self, state): pass
    assert tensor_bytes(MetadataScorer()) > 0
