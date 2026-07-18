from .augment import (
    CONDITION_FAMILIES,
    REGISTRY,
    Augment,
    Truncate,
    build_pipeline,
    subset,
)
from .bach10synth import PitchDatasetBach10Synth
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
from .m4singer import PitchDatasetM4Singer
from .mdb import PitchDatasetMDBStemSynth
from .mir1k import PitchDatasetMIR1K
from .nsynth import PitchDatasetNSynth
from .speechsynth import PitchDatasetSpeechSynth
from .urmp import PitchDatasetURMP
from .vocadito import PitchDatasetVocadito

__all__ = [
    "CONDITION_FAMILIES",
    "REGISTRY",
    "Augment",
    "PitchDataset",
    "PitchDatasetAPLAWD",
    "PitchDatasetAVID",
    "PitchDatasetBach10Synth",
    "PitchDatasetCMUArctic",
    "PitchDatasetFDA",
    "PitchDatasetKEELE",
    "PitchDatasetM4Singer",
    "PitchDatasetMDBStemSynth",
    "PitchDatasetMIR1K",
    "PitchDatasetMOCHA",
    "PitchDatasetNSynth",
    "PitchDatasetOSFGlottis",
    "PitchDatasetPTDB",
    "PitchDatasetSVD",
    "PitchDatasetSpeechSynth",
    "PitchDatasetURMP",
    "PitchDatasetVocadito",
    "Truncate",
    "build_pipeline",
    "subset",
]

# Separate registries for different dataset types/capabilities
_PITCH_REGISTRY = {
    "PTDB": PitchDatasetPTDB,
    "NSynth": PitchDatasetNSynth,
    "MDBStemSynth": PitchDatasetMDBStemSynth,
    "SpeechSynth": PitchDatasetSpeechSynth,
    "MIR1K": PitchDatasetMIR1K,
    # M4Singer: score-grade GT (voicing-reliable, pitch is intended-score); selection/voicing use.
    "M4Singer": PitchDatasetM4Singer,
    "Vocadito": PitchDatasetVocadito,
    "Bach10Synth": PitchDatasetBach10Synth,
    # EGG (laryngograph) speech corpora: consensus f0 (committed npz) by default; PTDB/KEELE/FDA
    # additionally offer the dataset authors' shipped reference (label_source="reference").
    "MOCHA": PitchDatasetMOCHA,
    "CMUArctic": PitchDatasetCMUArctic,
    "AVID": PitchDatasetAVID,
    "OSFGlottis": PitchDatasetOSFGlottis,
    "SVD": PitchDatasetSVD,
    "APLAWD": PitchDatasetAPLAWD,
    "KEELE": PitchDatasetKEELE,
    "FDA": PitchDatasetFDA,
    # multi-instrument music with gold frame-level f0
    "URMP": PitchDatasetURMP,
}

def get_pitch_dataset(name: str):
    """
    Get a pitch dataset class by name from the pitch registry.

    Args:
        name (str): Dataset name (case-sensitive)

    Returns:
        Type[PitchDataset]: PitchDataset class from the registry

    Raises:
        ValueError: If dataset name is not found in the pitch registry.

    Example:
        >>> dataset_cls = get_pitch_dataset("PTDB")
        >>> dataset = dataset_cls(root_dir="./data/PTDB", sample_rate=22050, hop_size=256)
    """
    if name not in _PITCH_REGISTRY:
        raise ValueError(
            f"Unknown pitch dataset: {name}. Available: {list(_PITCH_REGISTRY.keys())}"
        )
    return _PITCH_REGISTRY[name]


def register_dataset(name: str, dataset_class):
    """
    Register a new dataset class in the pitch registry.

    Args:
        name (str): Name to register the dataset under.
        dataset_class: Dataset class to register (must inherit from PitchDataset).

    Raises:
        TypeError: If dataset_class does not inherit from PitchDataset.

    Example:
        >>> class MyCustomPitchDataset(PitchDataset):
        ...     pass
        >>> register_dataset("CustomPitch", MyCustomPitchDataset)
    """
    if not issubclass(dataset_class, PitchDataset):
        raise TypeError(
            f"Dataset class must inherit from PitchDataset, got {dataset_class}"
        )
    _PITCH_REGISTRY[name] = dataset_class


def list_pitch_datasets() -> list[str]:
    """
    List all available pitch dataset names in the pitch registry.

    Returns:
        List[str]: List of registered pitch dataset names.

    Example:
        >>> print(list_pitch_datasets())
        ['PTDB', 'NSynth', 'MDBStemSynth', 'SpeechSynth', 'MIR1K', 'Vocadito', 'Bach10Synth']
    """
    return list(_PITCH_REGISTRY.keys())


def list_note_datasets() -> list[str]:
    """
    List the pitch datasets that also carry ground-truth notes (PitchDataset.provides_notes).

    Notes are a CAPABILITY of a pitch dataset, not a separate type, so this is the one pitch
    registry filtered by the capability flag, never a second registry to keep in sync. Whether a
    note-capable dataset belongs on the note LEADERBOARD is a separate policy (e.g. M4Singer has
    notes but is excluded for score-grade GT); that decision lives in the runner, not here.

    Returns:
        List[str]: Registered dataset names whose samples include a "notes" key.
    """
    return [name for name, cls in _PITCH_REGISTRY.items() if getattr(cls, "provides_notes", False)]


