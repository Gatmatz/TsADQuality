from tsadquality.enums.EasilyStringifyableEnum import EasilyStringifyableEnum


class DetectorModel(EasilyStringifyableEnum):
    LOF = "LocalOutlierFactor"
    ISO = "IsolationForest"
    MP = "MatrixProfile"
    AutoEncoder = "AutoEncoder"


DETECTORS = [
    DetectorModel.LOF,
    DetectorModel.ISO,
    DetectorModel.MP,
    DetectorModel.AutoEncoder
]