from mindset.methods.joint_one_price import OnePriceJointMemory
from mindset.methods.joint import JointMindsetMemory
from mindset.methods.compact import CompactConfig
from test_compact import Encoder, Reranker, event


def test_one_price_preserves_input_config_and_existing_price():
    config = CompactConfig(complexity_penalty=.1, switch_margin=.05)
    memory = OnePriceJointMemory(config, encoder=Encoder(), local_reranker=Reranker())
    assert config.switch_margin == .05
    assert memory.config.switch_margin == 0. and memory.config.complexity_penalty == .1
    assert memory.telemetry()["removed_original_switch_margin"] == .05


def test_not_globally_equivalent_switch_cost_can_matter():
    # A deliberately large margin is useful for preventing an immediate version.
    # The new candidate MUST NOT assert global equivalence from original100.
    config = CompactConfig(complexity_penalty=1., switch_margin=1.)
    baseline = JointMindsetMemory(config, encoder=Encoder(), local_reranker=Reranker())
    candidate = OnePriceJointMemory(config, encoder=Encoder(), local_reranker=Reranker())
    events = [event(1, "I live in Paris."), event(2, "I moved to Rome.")]
    baseline.ingest(events); candidate.ingest(events)
    assert baseline.operation_log.all()[-1].operation == "reinforce"
    assert candidate.operation_log.all()[-1].operation == "supersede"
