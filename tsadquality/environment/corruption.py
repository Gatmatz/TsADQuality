import ast
import os

from dotenv import load_dotenv

from tsadquality.enums.data import CorruptionType

load_dotenv()


def _parse_list(name: str) -> list:
    raw = os.getenv(name)
    if not raw:
        return []
    return ast.literal_eval(raw)


WHITE_NOISE = _parse_list("WHITE_NOISE")

SPIKES_AMP = _parse_list("SPIKES_AMP")
SPIKES_RATE = _parse_list("SPIKES_RATE")

STUCK_RATE = _parse_list("STUCK_RATE")
STUCK_BLOCKS = _parse_list("STUCK_BLOCKS")

POINT_SWAP_RATE = _parse_list("POINT_SWAP_RATE")

SEGMENT_SWAP_RATE = _parse_list("SEGMENT_SWAP_RATE")
SEGMENT_SWAP_NUM_SWAPS = _parse_list("SEGMENT_SWAP_NUM_SWAPS")

PERMUTATION_NUM = _parse_list("PERMUTATION_NUM")

MISSING_POINT_RATE = _parse_list("MISSING_POINT_RATE")
MISSING_POINT_MECH = _parse_list("MISSING_POINT_MECH")

MISSING_BURST_RATE = _parse_list("MISSING_BURST_RATE")
MISSING_BURST_NUM = _parse_list("MISSING_BURST_NUM")

MISSING_GE_A = _parse_list("MISSING_GE_A")
MISSING_GE_B = _parse_list("MISSING_GE_B")

# Maps each CorruptionType to the injector kwarg(s) swept from its .env setting(s).
CORRUPTION_SETTINGS = {
    CorruptionType.NOISE_SNR: {"snr_db": WHITE_NOISE},
    CorruptionType.SPIKE: {"multiplier": SPIKES_AMP, "fraction": SPIKES_RATE},
    CorruptionType.STUCK: {"fraction": STUCK_RATE, "stuck_blocks": STUCK_BLOCKS},
    CorruptionType.POINT_SWAP: {"fraction": POINT_SWAP_RATE},
    CorruptionType.SEGMENT_SWAP: {
        "fraction": SEGMENT_SWAP_RATE,
        "num_swaps": SEGMENT_SWAP_NUM_SWAPS,
    },
    CorruptionType.POINT_MISSING: {
        "fraction": MISSING_POINT_RATE,
        "mechanism": MISSING_POINT_MECH,
    },
    CorruptionType.BURST_MISSING: {
        "fraction": MISSING_BURST_RATE,
        "num_bursts": MISSING_BURST_NUM,
    },
    CorruptionType.GILBERT_ELLIOTT: {
        "a": MISSING_GE_A,
        "b": MISSING_GE_B,
    },
    CorruptionType.PERMUTATION_SWAP: {"num_permutations": PERMUTATION_NUM},
}
