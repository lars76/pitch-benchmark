from .base import PitchDataset
from .ptdb import PitchDatasetPTDB
from .mdb import PitchDatasetMDBStemSynth
from .nsynth import PitchDatasetNSynth
from .speechsynth import PitchDatasetSpeechSynth
from .transforms import NoiseAugmentedDataset, CHiMeNoiseDataset
from .utils import ChunkedPitchDataset, pitch_collate_fn

__all__ = [
    "PitchDataset",
    "PitchDatasetPTDB",
    "PitchDatasetMDBStemSynth",
    "PitchDatasetNSynth",
    "PitchDatasetSpeechSynth",
    "NoiseAugmentedDataset",
    "CHiMeNoiseDataset",
    "ChunkedPitchDataset",
    "pitch_collate_fn",
]

_REGISTRY = {
    "PTDB": PitchDatasetPTDB,
    "NSynth": PitchDatasetNSynth,
    "MDBStemSynth": PitchDatasetMDBStemSynth,
    "SpeechSynth": PitchDatasetSpeechSynth,
}


def get_dataset(name: str):
    """
    Get dataset class by name from the registry.

    Args:
        name (str): Dataset name (case-sensitive)

    Returns:
        Dataset class from the registry

    Raises:
        ValueError: If dataset name is not found in registry

    Example:
        >>> dataset_cls = get_dataset("PTDB")
        >>> dataset = dataset_cls(sample_rate=22050, hop_size=256)
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def register_dataset(name: str, dataset_class):
    """
    Register a new dataset class in the registry.

    Args:
        name (str): Name to register the dataset under
        dataset_class: Dataset class (should inherit from PitchDataset)

    Example:
        >>> class MyCustomDataset(PitchDataset):
        ...     pass
        >>> register_dataset("Custom", MyCustomDataset)
    """
    if not issubclass(dataset_class, PitchDataset):
        raise TypeError(
            f"Dataset class must inherit from PitchDataset, got {dataset_class}"
        )
    _REGISTRY[name] = dataset_class


def list_datasets():
    """
    List all available dataset names in the registry.

    Returns:
        List[str]: List of registered dataset names

    Example:
        >>> print(list_datasets())
        ['PTDB', 'NSynth', 'MDBStemSynth', 'SpeechSynth']
    """
    return list(_REGISTRY.keys())
