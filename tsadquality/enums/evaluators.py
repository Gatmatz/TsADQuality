from tsadquality.enums.EasilyStringifyableEnum import EasilyStringifyableEnum

class EvaluationMethod(EasilyStringifyableEnum):
    """Enum for evaluation methods."""

    AUC_ROC = "AUC_ROC"
    AUC_PR = "AUC_PR"
    PRECISION_3SIGMA = "PRECISION_3SIGMA"

    def get_class(self):
        match self:
            case EvaluationMethod.AUC_ROC:
                from tsadquality.evaluators import AUCROCEvaluator

                return AUCROCEvaluator
            case EvaluationMethod.AUC_PR:
                from tsadquality.evaluators import AUCPREvaluator

                return AUCPREvaluator
            case EvaluationMethod.PRECISION_3SIGMA:
                from tsadquality.evaluators import ThreeSigmaPrecisionEvaluator

                return ThreeSigmaPrecisionEvaluator
