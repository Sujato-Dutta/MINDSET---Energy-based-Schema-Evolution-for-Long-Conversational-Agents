import importlib.util
from pathlib import Path
import pytest
from test_compact import Encoder,Reranker,event
from mindset.methods.compact import CompactConfig
from mindset.methods.joint import JointMindsetMemory
from mindset.acl.clients import Completion

spec=importlib.util.spec_from_file_location('causal_probe',Path(__file__).parents[1]/'scripts/test_mindset_causality.py')
probe=importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)

@pytest.mark.parametrize('cls',[JointMindsetMemory,probe.RuleMemory,probe.NearestMergeMemory])
def test_controller_controls_preserve_all_sources(cls):
    m=cls(CompactConfig(complexity_penalty=.1,switch_margin=.05),encoder=Encoder(),local_reranker=Reranker())
    m.ingest([event(1,'I live in Paris.'),event(2,'I moved to Rome.'),event(3,'I live in Rome.')])
    assert m.validate_invariants()['grounded_episode_count']==3
    assert len(probe.flat_pool(m,'Where does Sam live?'))==3

def test_nearest_merge_really_disables_temporal_versions():
    m=probe.NearestMergeMemory(encoder=Encoder(),local_reranker=Reranker())
    m.ingest([event(1,'I live in Paris.'),event(2,'I moved to Rome.')])
    assert len(m.schemas)==1 and m.operation_log.all()[-1].operation=='reinforce'
    assert not m._decision_certificates

def test_rule_certificate_does_not_claim_minimum_energy():
    m=probe.RuleMemory(CompactConfig(complexity_penalty=1.,switch_margin=1.),encoder=Encoder(),local_reranker=Reranker())
    m.ingest([event(1,'I live in Paris.'),event(2,'I moved to Rome.')])
    cert=m._decision_certificates['e2']
    assert cert['policy']=='fixed_rule_control'
    assert cert['optimality_gap']>0

def test_causal_native_qa_does_not_invent_lme_judge_accuracy():
    spec=importlib.util.spec_from_file_location('causal_answer',Path(__file__).parents[1]/'scripts/answer_causal_probe.py')
    answer=importlib.util.module_from_spec(spec); spec.loader.exec_module(answer)
    result=Completion('Paris',10,2,.1,'id')
    row=answer.score_prediction({'dataset':'longmemeval','category':'knowledge-update','answer':'Paris'},result)
    assert row['native_f1'] is None and row['native_judge_accuracy'] is None
    assert row['secondary_token_f1']==1 and row['judge_status']=='not_judged'
