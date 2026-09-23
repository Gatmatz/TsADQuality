"""Shared plot styling constants, so every plot in `tsadquality.plots` looks consistent."""

from tsadquality.enums.detectors import DetectorModel

DEFAULT_METRIC = "AUC-ROC"

FONT_FAMILY = "Arial"

# Paul Tol's "bright" qualitative palette (colorblind-safe), assigned in a fixed
# order. Green is avoided even though it's part of the palette, since it reads as
# "good"/"correct" and would bias a detector's line rather than just identifying it.
DETECTOR_COLORS: dict[DetectorModel, str] = {
    DetectorModel.LOF: "#4477AA",
    DetectorModel.ISO: "#EE6677",
    DetectorModel.MP: "#66CCEE",
    DetectorModel.AutoEncoder: "#AA3377",
}
DETECTOR_MARKERS: dict[DetectorModel, str] = {
    DetectorModel.LOF: "o",
    DetectorModel.ISO: "s",
    DetectorModel.MP: "^",
    DetectorModel.AutoEncoder: "D",
}

# Human-readable names for plot labels/legends, since `DetectorModel`'s own values
# (e.g. "MatrixProfile") are identifiers, not display text.
DETECTOR_DISPLAY_NAMES: dict[DetectorModel, str] = {
    DetectorModel.LOF: "Local Outlier Factor",
    DetectorModel.ISO: "Isolation Forest",
    DetectorModel.MP: "Matrix Profile",
    DetectorModel.AutoEncoder: "AutoEncoder",
}
