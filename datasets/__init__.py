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
from .laryngograph import PitchDatasetPTDB
from .mdb import PitchDatasetMDBStemSynth
from .mir1k import PitchDatasetMIR1K
from .nsynth import PitchDatasetNSynth
from .speechsynth import PitchDatasetSpeechSynth
from .vocadito import PitchDatasetVocadito

__all__ = [
    "CONDITION_FAMILIES",
    "REGISTRY",
    "Augment",
    "PitchDataset",
    "PitchDatasetBach10Synth",
    "PitchDatasetMDBStemSynth",
    "PitchDatasetMIR1K",
    "PitchDatasetNSynth",
    "PitchDatasetPTDB",
    "PitchDatasetSpeechSynth",
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
    "Vocadito": PitchDatasetVocadito,
    "Bach10Synth": PitchDatasetBach10Synth,
}

_TRANSCRIPTION_REGISTRY = {
    "Vocadito": PitchDatasetVocadito,
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


def get_transcription_dataset(name: str):
    """
    Get a transcription dataset class by name from the transcription registry.

    Args:
        name (str): Dataset name (case-sensitive)

    Returns:
        Type[PitchDataset]: PitchDataset class from the registry

    Raises:
        ValueError: If dataset name is not found in the transcription registry,
                    or if the dataset does not inherit from PitchDataset.

    Example:
        >>> dataset_cls = get_transcription_dataset("Bach10Synth")
        >>> dataset = dataset_cls(root_dir="./data/Bach10Synth", sample_rate=22050, hop_size=256)
        >>> midi_data = dataset.get_midi_transcription(0)
    """
    if name not in _TRANSCRIPTION_REGISTRY:
        raise ValueError(
            f"Unknown transcription dataset: {name}. Available: {list(_TRANSCRIPTION_REGISTRY.keys())}"
        )
    dataset_class = _TRANSCRIPTION_REGISTRY[name]
    if not issubclass(dataset_class, PitchDataset):
        # Defensive: the registry should already guarantee this.
        raise TypeError(
            f"Registered dataset '{name}' for transcription does not inherit from PitchDataset."
        )
    return dataset_class


def register_dataset(name: str, dataset_class, dataset_type: str = "pitch"):
    """
    Register a new dataset class in the appropriate registry based on its type.

    Args:
        name (str): Name to register the dataset under.
        dataset_class: Dataset class to register.
        dataset_type (str): The type of dataset to register ("pitch" or "transcription").
                            Defaults to "pitch".

    Raises:
        TypeError: If the dataset_class does not inherit from the correct base class
                   (PitchDataset) for the specified type.
        ValueError: If an invalid dataset_type is provided.

    Example:
        >>> class MyCustomPitchDataset(PitchDataset):
        ...     pass
        >>> register_dataset("CustomPitch", MyCustomPitchDataset, "pitch")

        >>> class MyCustomTranscriptionDataset(PitchDataset):
        ...     def get_midi_transcription(self, index): return []
        >>> register_dataset("CustomTranscription", MyCustomTranscriptionDataset, "transcription")
    """
    if dataset_type == "pitch":
        if not issubclass(dataset_class, PitchDataset):
            raise TypeError(
                f"Dataset class for 'pitch' type must inherit from PitchDataset, got {dataset_class}"
            )
        _PITCH_REGISTRY[name] = dataset_class
    elif dataset_type == "transcription":
        if not issubclass(dataset_class, PitchDataset):
            raise TypeError(
                f"Dataset class for 'transcription' type must inherit from PitchDataset, got {dataset_class}"
            )
        _TRANSCRIPTION_REGISTRY[name] = dataset_class
    else:
        raise ValueError(
            f"Invalid dataset_type for registration: '{dataset_type}'. "
            "Must be 'pitch' or 'transcription'."
        )


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


def list_transcription_datasets() -> list[str]:
    """
    List all available transcription dataset names in the transcription registry.

    Returns:
        List[str]: List of registered transcription dataset names.

    Example:
        >>> print(list_transcription_datasets())
        ['Bach10Synth']
    """
    return list(_TRANSCRIPTION_REGISTRY.keys())
