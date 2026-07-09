"""Laryngograph speech datasets: corpora with electroglottograph-derived consensus f0 labels.

The family base (LaryngographSpeechDataset) and the shared energy gate live in base.py; each
concrete corpus is a thin subclass in its own file (set NAME + _discover, optionally
_load_author_labels). Add a new corpus by dropping a module here and re-exporting it below.
"""
from .base import LaryngographSpeechDataset, energy_voicing_gate
from .ptdb import PitchDatasetPTDB

__all__ = ["LaryngographSpeechDataset", "PitchDatasetPTDB", "energy_voicing_gate"]
