from enum import Enum


class EasilyStringifyableEnum(Enum):
    """
    Use this Enum class if you want to be able to call `str(your_enum_instance)`
    and produce the same as `your_enum_instance.value`.
    """

    def __str__(self):
        return self.value