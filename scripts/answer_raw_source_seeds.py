"""Matched answers for the original100 source-substrate and seed controls."""
from mindset.acl.matched_source_answers import run_panel
from mindset.acl.pipeline import ROOT

ARMS=('joint_ref','expanded_joint_ref','flat_ce_seed','hybrid_seed')
CONTRASTS=(('expanded_joint_ref','joint_ref'),('flat_ce_seed','expanded_joint_ref'),
           ('hybrid_seed','flat_ce_seed'),('hybrid_seed','joint_ref'))

if __name__=='__main__':
    prepared=ROOT/'outputs/acl/raw_source_seeds_r01'
    run_panel(prepared,prepared/'answers',ARMS,CONTRASTS,
              'original100 state-substrate/seed controls, not energy-only ablation or novelty claim',
              ROOT/'outputs/acl/semantic_witness_r01/answers')
