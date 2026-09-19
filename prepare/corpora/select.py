from pathlib import Path

import numpy as np
from torch.utils.data import Subset

from constants import DESIGN_SEED, HOP_SIZE, SALT_SAMPLING, SAMPLE_RATE
from corpora import get_pitch_dataset
from corpora.base import base_index, copy_eval_attrs
from seeds import corpus_uid

FRAME_KEYS = ("pitch", "periodicity", "pitch_conf")


class _Subset(Subset):

    def get_group(self, idx):
        return self.dataset.get_group(self.indices[idx])

    def base_index(self, idx):
        return base_index(self.dataset, self.indices[idx])


def subset(base_dataset, indices):
    sub = _Subset(base_dataset, list(indices))
    copy_eval_attrs(sub, base_dataset)
    return sub


class Segment:
    def __init__(self, base_dataset, seconds, frame_counts, skip_seconds=0.0):
        self.base_dataset = base_dataset
        self.seconds = float(seconds)
        self.skip_seconds = float(skip_seconds)
        copy_eval_attrs(self, base_dataset)
        self.n_frames = int(self.seconds * self.sample_rate / self.hop_size)
        if self.n_frames < 1:
            raise ValueError(
                f"Segment: {self.seconds}s is under one frame at {self.sample_rate} Hz / "
                f"hop {self.hop_size}"
            )
        counts = [int(c) for c in frame_counts]
        if len(counts) != len(base_dataset):
            raise ValueError(
                f"Segment: got {len(counts)} frame counts for {len(base_dataset)} items"
            )
        skip = int(self.skip_seconds * self.sample_rate / self.hop_size)
        self.windows = []
        for parent, count in enumerate(counts):
            usable = count - skip
            if usable < 1:
                continue
            n_full = usable // self.n_frames
            for w in range(n_full):
                self.windows.append((parent, skip + w * self.n_frames, self.n_frames))
            tail = usable - n_full * self.n_frames
            if tail > 0 and (n_full == 0 or tail >= self.n_frames // 2):
                self.windows.append((parent, skip + n_full * self.n_frames, tail))
        if not self.windows:
            raise ValueError(f"Segment: all {len(counts)} items are empty")

    def __len__(self):
        return len(self.windows)

    def get_group(self, idx):
        return self.base_dataset.get_group(self.windows[idx][0])

    def base_index(self, idx):
        return int(idx)

    def __getitem__(self, idx):
        parent, f0, length = self.windows[idx]
        sample = dict(self.base_dataset[parent])
        audio = sample["audio"]
        if audio.dim() > 1:
            audio = audio.squeeze(0)

        f1 = f0 + length
        s0, s1 = f0 * self.hop_size, f1 * self.hop_size
        if audio.numel() < s1:
            raise ValueError(
                f"Segment: item {parent} at frame {f0} needs {s1} samples, audio has "
                f"{audio.numel()} (frame count disagreed with the decoded audio)"
            )
        sample["audio"] = audio[s0:s1].clone()
        for key in FRAME_KEYS:
            if sample.get(key) is not None:
                if sample[key].numel() < f1:
                    raise ValueError(
                        f"Segment: item {parent} at frame {f0} needs {f1} frames of '{key}', "
                        f"has {sample[key].numel()}"
                    )
                sample[key] = sample[key][f0:f1].clone()
        if sample.get("wav_path") is not None:
            p = Path(str(sample["wav_path"]))
            sample["wav_path"] = p.with_name(f"{p.stem}#seg{f0:07d}{p.suffix}")
        return sample


def segmented(base, dataset, seconds):
    n_frames = int(seconds * base.sample_rate / base.hop_size)
    bound = getattr(base, "MAX_ITEM_SECONDS", None)
    skip = float(getattr(base, "SKIP_LEAD_SECONDS", 0.0) or 0.0)
    try:
        counts = [int(base.item_grid_frames(i)) for i in range(len(base))]
    except NotImplementedError as e:
        if bound is not None and float(bound) <= seconds:
            return base
        raise ValueError(
            f"{dataset}: cannot enumerate scored windows (no item_grid_frames) and declares no "
            f"MAX_ITEM_SECONDS <= {seconds}s. Freezing it would silently score only its first "
            f"{seconds}s, which is how AVID and OSFGlottis became noise. Implement "
            f"item_grid_frames, or declare a true MAX_ITEM_SECONDS."
        ) from e
    if max(counts) < n_frames:
        return base
    seg = Segment(base, seconds, counts, skip_seconds=skip)
    silent, keep = {}, []
    for i, (parent, f0, length) in enumerate(seg.windows):
        if parent not in silent:
            silent[parent] = np.asarray(base.item_grid_silent(parent), dtype=bool)
        mask = silent[parent][f0:f0 + length]
        if not (mask.size and mask.all()):
            keep.append(i)
    if not keep:
        raise ValueError(
            f"{dataset}: every one of {len(seg)} windows is digital silence"
        )
    return seg if len(keep) == len(seg) else subset(seg, keep)


def sampling_phase(dataset):
    return float(np.random.default_rng(
        np.random.SeedSequence([DESIGN_SEED, SALT_SAMPLING, corpus_uid(dataset)])).random())


def stride_indices(total, n, phase):
    if n >= total:
        return list(range(total))
    return sorted(min(total - 1, int((j + phase) * total / n)) for j in range(n))


def build_eval_source(dataset, data_dir, *, sample_rate=SAMPLE_RATE, hop_size=HOP_SIZE,
                      max_clips=None, max_seconds=None, **loader_kwargs):
    base = get_pitch_dataset(dataset)(
        root_dir=data_dir, sample_rate=sample_rate, hop_size=hop_size, **loader_kwargs)
    if max_seconds:
        base = segmented(base, dataset, float(max_seconds))
    if max_clips and max_clips < len(base):
        base = subset(base, stride_indices(len(base), max_clips, sampling_phase(dataset)))
    return base
