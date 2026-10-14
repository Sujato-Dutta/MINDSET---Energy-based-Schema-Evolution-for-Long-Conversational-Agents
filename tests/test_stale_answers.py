from mindset.evaluation.stale_answers import stale_proxy

REPORTS=[{'value':'Birch','tense':'nonpast','timestamp':'2025-01-01'},
         {'value':'Cedar','tense':'nonpast','timestamp':'2025-01-02'}]


def score(prediction,reference='Cedar',**kwargs):
    return stale_proxy(prediction,reference,query_mode=kwargs.get('query_mode','state_current'),
        bound_reports=kwargs.get('bound_reports',REPORTS),official_correct=kwargs.get('official_correct',False))


def test_real_wrong_judgement_and_exact_earlier_value_required():
    assert score('Birch.')['stale'] is True
    assert score('Birch',official_correct=None)['stale'] is None
    assert score('Birch',official_correct=True)['stale'] is None


def test_correct_current_is_classified_only_with_correct_official_label():
    assert score('Cedar',official_correct=True)['stale'] is False
    assert score('Cedar',official_correct=False)['stale'] is None


def test_historical_comparison_or_missing_binding_not_called_stale():
    assert score('Birch',query_mode='historical')['stale'] is None
    assert score('Birch',query_mode='report_compare')['stale'] is None
    assert score('Birch',bound_reports=[])['stale'] is None


def test_ref_does_not_match_latest_source_remains_unclassified():
    assert score('Birch','Maple')['stale'] is None


def test_paraphrase_negation_unknown_and_abstention_not_classified_as_correct():
    for prediction in ('Not Birch; Cedar.','I think it is Birch.','Unknown','No information available.'):
        assert score(prediction)['stale'] is None


def test_past_only_reports_not_current_world_ground_truth():
    assert score('Birch',bound_reports=[{'value':'Cedar','tense':'past'}])['stale'] is None


def test_unknown_or_unordered_times_cannot_establish_staleness():
    assert score('Birch',bound_reports=list(reversed(REPORTS)))['stale'] is None
    assert score('Birch',bound_reports=[{'value':'Cedar','tense':'nonpast'}])['stale'] is None
