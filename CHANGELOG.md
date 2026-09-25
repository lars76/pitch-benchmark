# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
A major version changes the data or the score, so its results are not comparable with the
previous one. A minor version re-runs trackers or changes how they are measured.

## [Unreleased]

## [2.1.0] - 2026-09-25

### Changed
- SwiftF0 runs with swift-f0 0.3.0. A frame whose best pitch lies outside the corpus's search range now gets confidence 0. Its pitch F1 rises from 0.778 to 0.781, and its ranking is unchanged. The other trackers keep their 2.0.0 results.
- Speed is measured with every tracker pinned to one CPU core and the thread limits set to 1. It divides the audio duration by the median CPU time over five rounds. The trackers run in a shuffled order in each round, and each round times calls for at least 2 s. Version 2.0.0 measured wall time with each library's default threads, so its speeds are not comparable. For example, SwiftF0 runs at 179.6× real time (416.6× in 2.0.0) and RMVPE at 13.6× (50.0×). RAPT is now the fastest tracker at 1276.6×.
- SwiftF0 and BasicPitch build their ONNX Runtime sessions with one thread when `OMP_NUM_THREADS=1`, as `run.py` and `speed.py` set it.

### Added
- The speed table in BENCHMARK.md shows the standard deviation over the five rounds.
- `speed.py` also records the wall time. The report names trackers whose CPU time is below 80% of their wall time, since they wait for their own threads. In this release that is CREPE.

## [2.0.0] - 2026-09-19

### Changed
- The benchmark scores 19 trackers on 10 corpora. Version 1 scored 12 trackers on 8 corpora. The new trackers are DIO, FCPE, HarmoF0, Harvest, PESTO, REAPER and SHS. The corpora are APLAWD, AVID, Bach10Synth, FDA, KEELE, OSFGlottis, SVD, SpeechSynth, URMP and Vocadito.
- A common prepared dataset on [Hugging Face](https://huggingface.co/datasets/lars1234/pitch-benchmark) holds separate training and evaluation corpora. Version 1 required users to download and process each corpus. [prepare/](prepare/README.md) rebuilds the dataset from the raw corpora.
- Confidence thresholds are selected on `valid/` and scored on `test/`, separated by speaker or recording group. Version 1 selected them on the clips used for ranking.
- The score is pitch F1 at 50 cents. Version 1 combined six metrics, including three exponential transforms with manually chosen constants.
- Every clip is tested in all eight combinations of background sound, reverberation and microphone filtering, plus clean audio. The score averages the eight conditions and the ten corpora equally. Version 1 added background noise at 10–30 dB SNR.
- Scores show 95% confidence intervals. Each tracker lists its significant wins and losses against the others, adjusted for all pairwise comparisons.
- The benchmark runs with `run.py`, `speed.py` and `report.py`, installed with uv from `pyproject.toml`. They replace `pitch_benchmark.py`, `speed_benchmark.py`, `generate_report.py` and `requirements.txt`.

### Removed
- `visualize_algorithms.py`.

## [1.0.0] - 2025-08-25

Released with the [arXiv paper](https://arxiv.org/abs/2508.18440). The v1 tag marks its final state from March 2026.

### Added
- The original benchmark: 12 trackers on 8 corpora (Bach10Synth, MDBStemSynth, MIR1K, NSynth, PTDB, PTDBNoisy, SpeechSynth and Vocadito). The score is a harmonic mean of six metrics, with background noise from CHiME-Home at 10–30 dB SNR. Wrappers for DIO and Harvest are included but not in the results.

[Unreleased]: https://github.com/lars76/pitch-benchmark/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/lars76/pitch-benchmark/compare/v2...v2.1.0
[2.0.0]: https://github.com/lars76/pitch-benchmark/compare/v1...v2
[1.0.0]: https://github.com/lars76/pitch-benchmark/releases/tag/v1
