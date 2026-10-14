from pathlib import Path
import importlib.util
import pytest


spec=importlib.util.spec_from_file_location('capacity_worker',Path(__file__).resolve().parents[1]/'scripts/check_local_capacity.py')
worker=importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def test_server_alias_accepts_only_logical_repository_or_exact_locked_snapshot():
    checkpoint={'checkpoint':'Example/Model','revision':'abc123'}
    assert worker.server_identifier({'data':[{'id':'Example/Model'}]},checkpoint,'/cache')=='Example/Model'
    exact='/cache/hub/models--Example--Model/snapshots/abc123'
    assert worker.server_identifier({'data':[{'id':exact}]},checkpoint,'/cache')==exact
    with pytest.raises(ValueError): worker.server_identifier({'data':[{'id':exact.replace('abc123','newrev')}]},checkpoint,'/cache')
    with pytest.raises(ValueError): worker.server_identifier({'data':[{'id':'Different/Model'}]},checkpoint,'/cache')
    with pytest.raises(ValueError): worker.server_identifier({'data':[{'id':'Example/Model'},{'id':exact}]},checkpoint,'/cache')
