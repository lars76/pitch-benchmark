from .annotated import (
    PitchDatasetBach10Synth,
    PitchDatasetMDBStemSynth,
    PitchDatasetNSynth,
    PitchDatasetURMP,
    PitchDatasetVocadito,
)
from .base import PitchDataset
from .laryngograph import (
    PitchDatasetAPLAWD,
    PitchDatasetAVID,
    PitchDatasetCMUArctic,
    PitchDatasetFDA,
    PitchDatasetKEELE,
    PitchDatasetMOCHA,
    PitchDatasetOSFGlottis,
    PitchDatasetPTDB,
    PitchDatasetSVD,
)
from .speechsynth import PitchDatasetSpeechSynth

_PITCH_REGISTRY = {
    "PTDB": PitchDatasetPTDB,
    "NSynth": PitchDatasetNSynth,
    "MDBStemSynth": PitchDatasetMDBStemSynth,
    "SpeechSynth": PitchDatasetSpeechSynth,
    "Vocadito": PitchDatasetVocadito,
    "Bach10Synth": PitchDatasetBach10Synth,
    "MOCHA": PitchDatasetMOCHA,
    "CMUArctic": PitchDatasetCMUArctic,
    "AVID": PitchDatasetAVID,
    "OSFGlottis": PitchDatasetOSFGlottis,
    "SVD": PitchDatasetSVD,
    "APLAWD": PitchDatasetAPLAWD,
    "KEELE": PitchDatasetKEELE,
    "FDA": PitchDatasetFDA,
    "URMP": PitchDatasetURMP,
}

EVAL = ("KEELE", "FDA", "APLAWD", "AVID", "OSFGlottis", "SVD", "SpeechSynth",
        "Vocadito", "URMP", "Bach10Synth")

TRAIN = {
    "MDBStemSynth": {},
    "NSynth": {"instrument_sources": ["acoustic"], "fmin": 1046.5, "fmax": 2093.75},
    "PTDB": {},
    "MOCHA": {},
    "CMUArctic": {},
}


def get_pitch_dataset(name):
    if name not in _PITCH_REGISTRY:
        raise ValueError(
            f"Unknown pitch dataset: {name}. Available: {list(_PITCH_REGISTRY.keys())}"
        )
    return _PITCH_REGISTRY[name]


def list_pitch_datasets():
    return list(_PITCH_REGISTRY.keys())
