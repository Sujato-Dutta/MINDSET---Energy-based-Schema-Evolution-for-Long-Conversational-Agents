"""Public MINDSET API. Revision v2 is not a separate baseline name."""

from mindset.methods.base import ConversationEvent, MemoryHit, MemoryMethod
from mindset.methods.nemori import NemoriFaithfulReproduction
from mindset.methods.compact import CompactConfig as MindsetConfig, CompactMindsetMemory


def load_calibrated_config():
    """Load verified paper/development settings, not illustrative class defaults."""
    from mindset.acl.pipeline import config
    from mindset.acl.calibration import validate_calibration
    cfg=config(); validate_calibration(cfg)
    params=dict(cfg["mindset"]); params.pop("profile",None)
    return MindsetConfig(**params)


class MindsetMemory(CompactMindsetMemory):
    """MINDSET; default construction uses the verified calibrated configuration.

    Passing an explicit configuration is supported for independent development
    and prespecified ablations; it is not automatically a frozen paper run.
    The earlier direct-module class is retained for historical replay only.
    """
    def __init__(self,config=None,**kwargs):
        super().__init__(load_calibrated_config() if config is None else config,**kwargs)

__all__ = [
    "ConversationEvent",
    "MemoryHit",
    "MemoryMethod",
    "NemoriFaithfulReproduction",
    "MindsetConfig",
    "MindsetMemory",
    "load_calibrated_config",
]
