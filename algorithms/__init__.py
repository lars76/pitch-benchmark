from importlib import import_module
from typing import Dict, List, Optional, Type

from .base import PitchAlgorithm

# Algorithm metadata - maps names to (module_name, class_name, required_packages)
_ALGORITHM_METADATA = {
    "CREPE": ("crepe", "CREPEPitchAlgorithm", ["crepe", "tensorflow"]),
    "PENN": ("penn", "PENNPitchAlgorithm", ["penn"]),
    "Praat": ("praat", "PraatPitchAlgorithm", ["praat-parselmouth"]),
    "RAPT": ("rapt", "RAPTPitchAlgorithm", ["pysptk"]),
    "SWIPE": ("swipe", "SWIPEPitchAlgorithm", ["pysptk"]),
    "TorchCREPE": (
        "torchcrepe",
        "TorchCREPEPitchAlgorithm",
        ["torchcrepe", "torch"],
    ),
    "YAAPT": ("yaapt", "YAAPTPitchAlgorithm", ["AMFM-decompy"]),
    "pYIN": ("pyin", "pYINPitchAlgorithm", ["librosa"]),
    "BasicPitch": ("basicpitch", "BasicPitchPitchAlgorithm", ["basic-pitch"]),
    "SwiftF0": ("swiftf0", "SwiftF0PitchAlgorithm", ["swift-f0"]),
    "SPICE": (
        "spice",
        "SPICEPitchAlgorithm",
        ["tensorflow", "tensorflow-hub"],
    ),
    "RMVPE": (
        "rmvpe",
        "RMVPEPitchAlgorithm",
        ["torch"],
    ),
    "DIO": ("dio", "DIOPitchAlgorithm", ["pyworld"]),
    "Harvest": ("harvest", "HarvestPitchAlgorithm", ["pyworld"]),
}

# The _REGISTRY now acts as a cache for lazily loaded algorithms.
_REGISTRY: Dict[str, Type[PitchAlgorithm]] = {}
_IMPORT_ERRORS: Dict[str, str] = {}


def get_algorithm(
    name: str, fail_silently: bool = False
) -> Optional[Type[PitchAlgorithm]]:
    """
    Get algorithm class by name using lazy loading.
    An algorithm's module is only imported the first time it is requested.
    """
    # 1. Check if the algorithm is already cached
    if name in _REGISTRY:
        return _REGISTRY[name]

    # 2. Check if the algorithm name is valid
    if name not in _ALGORITHM_METADATA:
        if fail_silently:
            return None
        raise ValueError(f"Unknown algorithm: {name}")

    # 3. If not cached, try to import it now (the "lazy" part)
    module_name, class_name, deps = _ALGORITHM_METADATA[name]
    try:
        module = import_module(f".{module_name}", package=__package__)
        cls = getattr(module, class_name)
        _REGISTRY[name] = cls  # Cache the successfully imported class
        return cls
    except ImportError as e:
        _IMPORT_ERRORS[name] = str(e)
        if fail_silently:
            return None
        raise ImportError(
            f"Algorithm '{name}' requires packages: {deps}\n"
            f"Error: {_IMPORT_ERRORS[name]}\n"
            f"See README for installation instructions."
        )
    except Exception as e:
        _IMPORT_ERRORS[name] = str(e)
        if fail_silently:
            return None
        raise e


def list_algorithms() -> List[str]:
    """
    Return a list of all possible algorithm names without importing them.
    This is fast and suitable for populating command-line choices.
    """
    return list(_ALGORITHM_METADATA.keys())


def get_available_algorithms() -> List[str]:
    """
    Actively tries to import all algorithms and returns a list of those that are available.
    This function is "eager" and should be used for diagnostic purposes.
    """
    available = []
    for name in _ALGORITHM_METADATA:
        if get_algorithm(name, fail_silently=True):
            available.append(name)
    return available


def get_algorithm_dependencies(name: str) -> list:
    """Get required packages for a specific algorithm."""
    if name not in _ALGORITHM_METADATA:
        raise ValueError(f"Unknown algorithm: {name}")
    return _ALGORITHM_METADATA[name][2]


def register_algorithm(name: str, algorithm_class: Type[PitchAlgorithm]):
    """Register a custom algorithm."""
    if not issubclass(algorithm_class, PitchAlgorithm):
        raise TypeError("Algorithm must subclass PitchAlgorithm")
    _REGISTRY[name] = algorithm_class
