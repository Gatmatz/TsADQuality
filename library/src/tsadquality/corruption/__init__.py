from .base import TARGETS, BaseCorruptor
from .gilbert_elliott import GilbertElliottInjector
from .missing import MECHANISMS, BurstMissingInjector, MissingValueInjector
from .noise import WhiteNoiseSNRInjector
from .permutation import PermutationInjector
from .spike import SpikeInjector
from .stuck import SensorStuckInjector
from .swap import SwapInjector

# Name used in `Corruptor.add()` -> (injector class, required params, optional params).
# Point and segment swap share SwapInjector: a point swap is a swap without `num_swaps`.
CORRUPTIONS = {
    "noise": (WhiteNoiseSNRInjector, ("snr_db",), ()),
    "spikes": (SpikeInjector, ("fraction", "multiplier"), ()),
    "stuck": (SensorStuckInjector, ("fraction", "stuck_blocks"), ()),
    "point_swap": (SwapInjector, ("fraction",), ()),
    "segment_swap": (SwapInjector, ("fraction", "num_swaps"), ()),
    "permutation": (PermutationInjector, ("num_permutations",), ()),
    "point_missing": (MissingValueInjector, ("fraction",), ("mechanism",)),
    "burst_missing": (BurstMissingInjector, ("fraction", "num_bursts"), ("mechanism",)),
    "gilbert_elliott": (GilbertElliottInjector, ("a", "b"), ()),
}

__all__ = [
    "CORRUPTIONS",
    "MECHANISMS",
    "TARGETS",
    "BaseCorruptor",
    "BurstMissingInjector",
    "GilbertElliottInjector",
    "MissingValueInjector",
    "PermutationInjector",
    "SensorStuckInjector",
    "SpikeInjector",
    "SwapInjector",
    "WhiteNoiseSNRInjector",
]
