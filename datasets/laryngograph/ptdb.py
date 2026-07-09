from pathlib import Path

import numpy as np
import torch

from .base import LaryngographSpeechDataset


class PitchDatasetPTDB(LaryngographSpeechDataset):
    """Pitch Tracking Database (PTDB-TUG) speech.

    Audio is the close-talk microphone (``{MALE,FEMALE}/MIC/mic_*.wav``). Ground-truth f0 defaults
    to the cross-family consensus labels (``datasets/laryngograph/PTDB.npz``); the
    label-method bias and laryngograph-bandpass silence artifacts of the shipped single-method
    reference are handled at generation time (see scripts/build_consensus_labels.py).

    ``label_source="author"`` instead loads the shipped RAPT-on-laryngograph reference
    (``REF/ref_*.f0``, column 0), with the mic-energy silence gate applied, for comparison.

    File-set policy (v2): every MIC/REF pair is included -- all 4718 utterances. The 347-file
    "noisy recordings" exclusion list used by earlier versions is deliberately gone: label quality
    is handled per frame by the consensus semantics instead (a voiced-but-pitch-uncertain frame
    gets f0=0, which drops it from RPA while keeping it in voicing F1), so whole files no longer
    need to be discarded for locally bad labels.
    """

    NAME = "PTDB"
    fmin = 65
    fmax = 300
    AUTHOR_LABEL_HOP_SECONDS = 0.01  # PTDB REF .f0 is a 10 ms-hop RAPT-on-laryngograph track
    # The REF track's frame i content sits ~21.7 ms AFTER the nominal i*10 ms stamp: half of
    # RAPT's 32 ms analysis window (16 ms) plus RAPT's forward NCCF offset (~6 ms, independently
    # measured on our own RAPT wrapper by tests/test_time_calibration.py). Measured by the
    # label-offset sweep with three calibrated reference trackers agreeing within 1 ms
    # (+21.7/+21.1/+22.1 ms; scripts/check_dataset_alignment.py, see TIMING.md).
    AUTHOR_LABEL_OFFSET_SECONDS = 0.0217

    def _discover(self) -> list[tuple[Path, str]]:
        items = []
        for gender in ("MALE", "FEMALE"):
            mic_dir = self.root_dir / gender / "MIC"
            if not mic_dir.exists():
                continue
            for wav in sorted(mic_dir.rglob("*.wav")):
                ref = Path(str(wav).replace("/MIC/", "/REF/")).with_name(
                    wav.name.replace("mic_", "ref_").replace(".wav", ".f0")
                )
                if ref.exists():
                    items.append((wav, wav.stem))
        return items

    def _load_author_labels(self, wav_path: Path, stem: str) -> tuple[torch.Tensor, torch.Tensor]:
        ref = Path(str(wav_path).replace("/MIC/", "/REF/")).with_name(
            wav_path.name.replace("mic_", "ref_").replace(".wav", ".f0")
        )
        try:
            pitch = torch.from_numpy(np.loadtxt(ref)[:, 0]).float()
        except Exception as e:
            raise OSError(f"Error loading F0 file {ref}: {e!s}") from e
        periodicity = (pitch > 0).float()
        return pitch, periodicity
