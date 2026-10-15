from mindset.acl.matched_source_answers import run_panel
from mindset.acl.pipeline import ROOT

ARMS = ('access_full', 'access_no_history', 'access_no_current', 'access_off', 'flat_expanded')
CONTRASTS = (('access_full', 'access_off'), ('access_full', 'access_no_history'),
             ('access_full', 'access_no_current'), ('access_full', 'flat_expanded'))

if __name__ == '__main__':
    prepared = ROOT / 'outputs/acl/accessibility_r01'
    run_panel(prepared, prepared / 'answers', ARMS, CONTRASTS,
              'integrated source-accessibility pilot; not novelty or paper confirmation; exact development100',
              ROOT / 'outputs/acl/raw_source_seeds_r01/answers')
