from dataclasses import asdict
import io
import json
import sqlite3
import pytest
from mindset.acl.clients import LocalClient,Completion
from mindset.acl.development_answers import inherit_cache
from mindset.acl.io import write_json


@pytest.mark.parametrize('content,finish',[('prefix','length'),('','stop'),(None,'length')])
def test_incomplete_diagnostic_keeps_same_request_and_usage_but_default_rejects(tmp_path,monkeypatch,content,finish):
    sent=[]
    def reply(request,**kwargs):
        sent.append(json.loads(request.data))
        return io.BytesIO(json.dumps({'choices':[{'message':{'content':content},'finish_reason':finish}],
            'usage':{'prompt_tokens':25,'completion_tokens':512},'id':'req'}).encode())
    monkeypatch.setattr('urllib.request.urlopen',reply)
    client=LocalClient('http://localhost/v1','frozen',revision='sha',cache_path=tmp_path/'cache.sqlite')
    prompt=[{'role':'user','content':'Question + raw evidence'}]
    with pytest.raises(ValueError,match='Empty or truncated'): client.complete(prompt,max_tokens=512,seed=0)
    assert client.cache.execute('SELECT count(*) FROM local_completions').fetchone()[0]==0
    client.allow_incomplete=True
    result=client.complete(prompt,max_tokens=512,seed=0)
    assert result.text==(content or '') and result.incomplete and result.finish_reason==finish
    assert result.input_tokens==25 and result.output_tokens==512
    assert sent[0]==sent[1] and sent[1]['max_tokens']==512 and sent[1]['seed']==0 and sent[1]['temperature']==0
    assert client.complete(prompt).cache_hit and len(sent)==2
    client.allow_incomplete=False
    with pytest.raises(ValueError,match='cached'): client.complete(prompt)


def test_malformed_answer_is_not_accepted_in_diagnostic(monkeypatch):
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:io.BytesIO(json.dumps({
        'choices':[{'message':{'content':['not text']},'finish_reason':'stop'}]}).encode()))
    client=LocalClient('http://localhost/v1','frozen'); client.allow_incomplete=True
    with pytest.raises(ValueError,match='Malformed'): client.complete([])


def source(tmp_path):
    old=tmp_path/'old'; new=tmp_path/'new'; old.mkdir(); new.mkdir()
    identity={key:'fixed' for key in ('prepared_sha256','manifest_sha256','calibration_sha256','checkpoint',
        'checkpoint_lock_sha256','prompt_hash','generation','methods','plan_sha256','serving')}
    identity['methods']=('joint','flat48','lightmem')
    write_json(old/'audit.json',identity); (old/'worker.lock').touch()
    cache=sqlite3.connect(old/'completion_cache.sqlite')
    cache.execute('CREATE TABLE local_completions (key TEXT PRIMARY KEY,payload TEXT)')
    cache.execute('INSERT INTO local_completions VALUES (?,?)',('request',json.dumps(asdict(Completion('yes',12,2,1,'req')))))
    cache.commit(); cache.close()
    return old,new,identity


def test_inherit_same_request_cache_without_modifying_source(tmp_path):
    old,new,identity=source(tmp_path); before=(old/'completion_cache.sqlite').read_bytes()
    inherit_cache(old,new,identity)
    assert (old/'completion_cache.sqlite').read_bytes()==before
    with sqlite3.connect(new/'completion_cache.sqlite') as cache:
        assert cache.execute('SELECT key FROM local_completions').fetchall()==[('request',)]
    inherit_cache(old,new,identity)  # Never overwrite an existing continuation cache.


def test_inherit_rejects_changed_request_identity(tmp_path):
    old,new,identity=source(tmp_path); identity['plan_sha256']='changed'
    with pytest.raises(ValueError,match='plan_sha256'): inherit_cache(old,new,identity)
    assert not (new/'completion_cache.sqlite').exists()


def test_explicit_cross_panel_cache_requires_same_model_prompt_and_generation(tmp_path):
    old,new,identity=source(tmp_path)
    identity.update(plan_sha256='new-declared-plan',prepared_sha256='new',methods=['retained_v1'])
    inherit_cache(old,new,identity,strict_plan=False)
    assert (new/'completion_cache.sqlite').exists()
    new=tmp_path/'wrong'; new.mkdir(); identity['checkpoint']='substituted'
    with pytest.raises(ValueError,match='checkpoint'): inherit_cache(old,new,identity,strict_plan=False)


def test_inherit_refuses_live_source_writer(tmp_path):
    import fcntl
    old,new,identity=source(tmp_path)
    with (old/'worker.lock').open('rb') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError): inherit_cache(old,new,identity)
    assert not (new/'completion_cache.sqlite').exists()


def test_inherit_rejects_incomplete_source_responses(tmp_path):
    old,new,identity=source(tmp_path)
    with sqlite3.connect(old/'completion_cache.sqlite') as cache:
        cache.execute('UPDATE local_completions SET payload=?',(json.dumps(asdict(Completion('',1,512,1,'req',incomplete=True))),))
    with pytest.raises(ValueError,match='incomplete'): inherit_cache(old,new,identity)
    assert not (new/'completion_cache.sqlite').exists()
