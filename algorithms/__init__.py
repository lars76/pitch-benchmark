from importlib import import_module

from .base import PitchAlgorithm

# Algorithm metadata - maps registry name to (module_name, class_name). The module name doubles as
# the pyproject optional-dependency extra (e.g. `uv sync --extra crepe`), so the packages each
# backend needs are declared in exactly one place: pyproject.toml [project.optional-dependencies].
# pYIN and RMVPE are covered by the core install and have no extra. tests/test_algorithm_extras.py
# enforces this module-name == extra-name correspondence.
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
}

# Trackers that need NO `uv sync --extra` -- there is no pip package to add. NOT a statement of
# importance: RMVPE's model is implemented in-repo (algorithms/rmvpe.py, torch + librosa only,
# weights downloaded at runtime) and pYIN is just librosa.pyin; both torch and librosa are core
# dependencies. Everything else ships an external backend package and thus a pyproject extra.
# Single source of truth for the "has an extra vs not" split; used by the install hint below and
# the conformance test (tests/test_algorithm_extras.py).
_NO_EXTRA_MODULES = frozenset({"pyin", "rmvpe"})

# _REGISTRY caches lazily loaded algorithm classes.
_REGISTRY: dict[str, type[PitchAlgorithm]] = {}
_IMPORT_ERRORS: dict[str, str] = {}


def get_algorithm(
    name: str, fail_silently: bool = False
) -> type[PitchAlgorithm] | None:
    """
    Get algorithm class by name using lazy loading.
    An algorithm's module is only imported the first time it is requested.
    """
    if name in _REGISTRY:
        return _REGISTRY[name]

    if name not in _ALGORITHM_METADATA:
        if fail_silently:
            return None
        raise ValueError(f"Unknown algorithm: {name}")

    # Not cached: import the module now (the lazy part).
    module_name, class_name = _ALGORITHM_METADATA[name]
    try:
        module = import_module(f".{module_name}", package=__package__)
        cls = getattr(module, class_name)
        _REGISTRY[name] = cls
        return cls
    except ImportError as e:
        _IMPORT_ERRORS[name] = str(e)
        if fail_silently:
            return None
        hint = (
            "reinstall the base environment with `uv sync`"
            if module_name in _NO_EXTRA_MODULES
            else f"add it with `uv sync --extra {module_name}`"
        )
        raise ImportError(
            f"Algorithm '{name}' is not available; {hint}.\n"
            f"(import failed: {_IMPORT_ERRORS[name]})"
        ) from e
    except Exception as e:
        _IMPORT_ERRORS[name] = str(e)
        if fail_silently:
            return None
        raise e


def build_algorithm(algo, sample_rate, hop_size, fmin, fmax, device="cpu"):
    """Construct a tracker instance, passing ``device=`` iff the class is device-aware.

    Single home for the construction pattern (was duplicated across the benchmark scripts).
    ``algo`` is an algorithm name (str) or an already-resolved class.
    """
    cls = get_algorithm(algo) if isinstance(algo, str) else algo
    kwargs = {"sample_rate": sample_rate, "hop_size": hop_size, "fmin": fmin, "fmax": fmax}
    if cls.device_backend is not None:
        kwargs["device"] = device
    return cls(**kwargs)


def list_algorithms() -> list[str]:
    """
    Return a list of all possible algorithm names without importing them.
    This is fast and suitable for populating command-line choices.
    """
    return list(_ALGORITHM_METADATA.keys())


def get_available_algorithms() -> list[str]:
    """
    Actively tries to import all algorithms and returns a list of those that are available.
    This function is "eager" and should be used for diagnostic purposes.
    """
    available = []
    for name in _ALGORITHM_METADATA:
        if get_algorithm(name, fail_silently=True):
            available.append(name)
    return available


def resolve_requested_algorithms(requested=None, *, report_skipped=False, on_empty=None):
    """Resolve a requested algorithm-name list to a concrete list, shared by the benchmark scripts.

    ``requested`` falsy (None / empty) -> every installed algorithm, so a partial install
    (``uv sync --extra ...``) just works; an explicit list is returned unchanged (a named-but-
    uninstalled backend surfaces later in get_algorithm). ``report_skipped`` prints which known
    algorithms are skipped because their backend is not installed. ``on_empty``, if given, is called
    with an error message when no algorithm resolves (pass ``parser.error`` to abort the CLI).
    """
    if requested:
        return list(requested)
    available = get_available_algorithms()
    if report_skipped:
        skipped = [a for a in list_algorithms() if a not in available]
        if skipped:
            # Hint with a real extra: the MODULE name is the extra name (not the display name, so
            # no .lower() of e.g. "pYIN"), and the _NO_EXTRA_MODULES backends have no extra at all.
            extras = [
                _ALGORITHM_METADATA[a][0]
                for a in skipped
                if _ALGORITHM_METADATA[a][0] not in _NO_EXTRA_MODULES
            ]
            hint = (
                f"install with e.g. `uv sync --extra {extras[0]}`"
                if extras
                else "reinstall the base environment with `uv sync`"
            )
            print(f"Skipping uninstalled algorithms: {', '.join(skipped)} ({hint})")
    if not available and on_empty is not None:
        on_empty(
            "No algorithm backends installed. Run `uv sync --all-extras` or "
            "`uv sync --extra <name>`."
        )
    return available


def register_algorithm(name: str, algorithm_class: type[PitchAlgorithm]):
    """Register a custom algorithm."""
    if not issubclass(algorithm_class, PitchAlgorithm):
        raise TypeError("Algorithm must subclass PitchAlgorithm")
    _REGISTRY[name] = algorithm_class
