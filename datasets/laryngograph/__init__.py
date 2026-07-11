"""Laryngograph speech datasets: corpora recorded with an electroglottograph (EGG).

The family base (LaryngographSpeechDataset) and the shared energy gate live in base.py. Each concrete
corpus implements the per-corpus reader for its ORIGINAL extracted download (``_iter_originals`` +
``_read_original``), which is reused by scripts/build_consensus_labels.py so the on-disk format lives
in one place. Ground truth is the committed cross-family ``consensus`` npz by default; PTDB/KEELE/FDA
additionally ship the dataset authors' f0 (``reference`` mode). Add a new corpus by dropping a module
here (with its reader) and re-exporting it below.
"""
from .aplawd import PitchDatasetAPLAWD
from .avid import PitchDatasetAVID
from .base import LaryngographSpeechDataset, energy_voicing_gate
from .cmuarctic import PitchDatasetCMUArctic
from .fda import PitchDatasetFDA
from .keele import PitchDatasetKEELE
from .mocha import PitchDatasetMOCHA
from .osf_glottis import PitchDatasetOSFGlottis
from .ptdb import PitchDatasetPTDB
from .svd import PitchDatasetSVD

__all__ = [
    "LaryngographSpeechDataset",
    "PitchDatasetAPLAWD",
    "PitchDatasetAVID",
    "PitchDatasetCMUArctic",
    "PitchDatasetFDA",
    "PitchDatasetKEELE",
    "PitchDatasetMOCHA",
    "PitchDatasetOSFGlottis",
    "PitchDatasetPTDB",
    "PitchDatasetSVD",
    "energy_voicing_gate",
]
