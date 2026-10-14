import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('regression_analysis',Path(__file__).resolve().parents[1]/'scripts/analyze_legacy_regression.py')
worker=importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def test_paired_uses_shared_ids_not_unmatched_method_averages():
    result=worker.paired({'q1':1.,'q2':0.,'extra':1.},{'q1':0.,'q2':1.})
    assert result['questions']==2 and result['mean_difference']==0.
    assert (result['wins'],result['ties'],result['losses'])==(1,0,1)
    assert result['first_mean']==result['second_mean']==.5


def test_empty_pair_is_rejected():
    with pytest.raises(ValueError): worker.paired({'x':1.},{'y':0.})


def test_partial_artifact_cannot_be_reported_as_complete():
    with pytest.raises(ValueError,match='complete'): worker.analyze([], [{'id':'q'}], [])


def test_duplicate_cells_are_rejected():
    with pytest.raises(ValueError,match='Duplicate'): worker.analyze([{'id':'q'},{'id':'q'}],[{'id':'q'}],[])
