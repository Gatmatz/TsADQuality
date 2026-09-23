from tsadquality.enums.EasilyStringifyableEnum import EasilyStringifyableEnum
from tsadquality.corruption.BaseError import BaseCorruptor


class CorruptionType(EasilyStringifyableEnum):
    # Missing
    POINT_MISSING = "PMIS"
    BURST_MISSING = "BMS"
    GILBERT_ELLIOTT = "GE"

    # White Noise
    NOISE_SNR = "SNR"

    # Spikes
    SPIKE = "SPK"

    # Sensor Freeze
    STUCK = "STK"

    # Swap
    POINT_SWAP = "PSW"
    SEGMENT_SWAP = "SSW"
    PERMUTATION_SWAP = "PMS"

    def get_class(self) -> BaseCorruptor.__class__:
        match self:
            case CorruptionType.POINT_MISSING:
                from tsadquality.corruption import MissingValueInjector

                return MissingValueInjector
            case CorruptionType.BURST_MISSING:
                from tsadquality.corruption import BurstMissingInjector

                return BurstMissingInjector
            case CorruptionType.GILBERT_ELLIOTT:
                from tsadquality.corruption import GilbertElliottInjector

                return GilbertElliottInjector
            case CorruptionType.NOISE_SNR:
                from tsadquality.corruption import WhiteNoiseSNRInjector

                return WhiteNoiseSNRInjector
            case CorruptionType.SPIKE:
                from tsadquality.corruption import SpikeInjector

                return SpikeInjector
            case CorruptionType.STUCK:
                from tsadquality.corruption import SensorStuckInjector

                return SensorStuckInjector
            case CorruptionType.POINT_SWAP:
                from tsadquality.corruption import SwapInjector

                return SwapInjector
            case CorruptionType.SEGMENT_SWAP:
                from tsadquality.corruption import SwapInjector

                return SwapInjector
            case CorruptionType.PERMUTATION_SWAP:
                from tsadquality.corruption import PermutationInjector

                return PermutationInjector


class DataPerfectness(EasilyStringifyableEnum):
    PERFECT = "PERF"
    IMPERFECT = "IMP"