from importlib import import_module

from .base import PitchAlgorithm

_ALGORITHM_METADATA = {
    "CREPE": ("crepe", "CREPEPitchAlgorithm"),
    "PENN": ("penn", "PENNPitchAlgorithm"),
    "Praat": ("praat", "PraatPitchAlgorithm"),
    "RAPT": ("rapt", "RAPTPitchAlgorithm"),
    "REAPER": ("reaper", "REAPERPitchAlgorithm"),
    "SWIPE": ("swipe", "SWIPEPitchAlgorithm"),
    "TorchCREPE": ("torchcrepe", "TorchCREPEPitchAlgorithm"),
    "YAAPT": ("yaapt", "YAAPTPitchAlgorithm"),
    "pYIN": ("pyin", "pYINPitchAlgorithm"),
    "BasicPitch": ("basicpitch", "BasicPitchPitchAlgorithm"),
    "SwiftF0": ("swiftf0", "SwiftF0PitchAlgorithm"),
    "SPICE": ("spice", "SPICEPitchAlgorithm"),
    "RMVPE": ("rmvpe", "RMVPEPitchAlgorithm"),
    "DIO": ("dio", "DIOPitchAlgorithm"),
    "Harvest": ("harvest", "HarvestPitchAlgorithm"),
    "HarmoF0": ("harmof0", "HarmoF0PitchAlgorithm"),
    "PESTO": ("pesto", "PESTOPitchAlgorithm"),
    "FCPE": ("fcpe", "FCPEPitchAlgorithm"),
    "SHS": ("shs", "SHSPitchAlgorithm"),
}

_NO_EXTRA_MODULES = frozenset({"pyin", "rmvpe", "harmof0"})

_REGISTRY = {}


def get_algorithm(name, fail_silently=False):
    if name in _REGISTRY:
        return _REGISTRY[name]

    if name not in _ALGORITHM_METADATA:
        if fail_silently:
            return None
        raise ValueError(f"Unknown algorithm: {name}")

    module_name, class_name = _ALGORITHM_METADATA[name]
    try:
        module = import_module(f".{module_name}", package=__package__)
        cls = getattr(module, class_name)
        _REGISTRY[name] = cls
        return cls
    except ImportError as e:
        if fail_silently:
            return None
        hint = (
            "reinstall the base environment with `uv sync`"
            if module_name in _NO_EXTRA_MODULES
            else f"add it with `uv sync --extra {module_name}`"
        )
        raise ImportError(
            f"Algorithm '{name}' is not available; {hint}.\n(import failed: {e})"
        ) from e
    except Exception:
        if fail_silently:
            return None
        raise


def build_algorithm(algo, sample_rate, hop_size, fmin, fmax):
    cls = get_algorithm(algo) if isinstance(algo, str) else algo
    return cls(sample_rate=sample_rate, hop_size=hop_size, fmin=fmin, fmax=fmax)


def list_algorithms():
    return list(_ALGORITHM_METADATA.keys())
