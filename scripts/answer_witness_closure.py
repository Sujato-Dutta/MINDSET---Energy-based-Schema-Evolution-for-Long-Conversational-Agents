from prepare_witness_closure import ARMS, SOURCE_ADAPTER
from mindset.acl.matched_source_answers import run_panel
from mindset.acl.witness_closure_answers import transform_plan
from mindset.acl.pipeline import ROOT

CONTRASTS = (('witness_closed', 'raw'), ('witness_closed', 'calendar_only'),
             ('witness_closed', 'closure_only'), ('witness_closed', 'window_calendar'))

if __name__ == '__main__':
    prepared = ROOT / 'outputs/acl/witness_closure_r01'
    run_panel(prepared, prepared / 'answers', ARMS, CONTRASTS,
        'One integrated witness-closure reconstruction candidate, matched controls, exposed original100',
        ROOT / 'outputs/acl/raw_source_seeds_r01/answers',
        source_adapter=SOURCE_ADAPTER, plan_transform=transform_plan)
