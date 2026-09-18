from tsadquality.enums.EasilyStringifyableEnum import EasilyStringifyableEnum


class DetectorModel(EasilyStringifyableEnum):
    LOF = "LocalOutlierFactor"
    ISO = "IsolationForest"
    MP = "MatrixProfile"
    AutoEncoder = "AutoEncoder"

    def get_class(self):
        match self:
            case DetectorModel.LOF:
                from tsadquality.detectors import LOFDetector

                return LOFDetector
            case DetectorModel.ISO:
                from tsadquality.detectors import IsolationForestDetector

                return IsolationForestDetector
            case DetectorModel.MP:
                from tsadquality.detectors import MatrixProfileDetector

                return MatrixProfileDetector
            case DetectorModel.AutoEncoder:
                from tsadquality.detectors import AutoEncoderDetector

                return AutoEncoderDetector


DETECTORS = [
    DetectorModel.LOF,
    DetectorModel.ISO,
    DetectorModel.MP,
    DetectorModel.AutoEncoder
]