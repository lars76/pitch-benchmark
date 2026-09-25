# Pitch Detection Benchmark

This is version 2.1 of the benchmark: 19 monophonic pitch trackers over 10 corpora, with a common prepared dataset. Each clip has nine versions:
clean audio and eight combinations of background sound, reverberation and microphone filtering.
[BENCHMARK.md](BENCHMARK.md) contains the full report.

## Results

| Tracker | Score: pitch F1@50c ↑ [95% CI] | Speed (× real time, one core) ↑ | Beats ↑ | Loses to ↓ | Undetermined |
|---|---:|---:|---:|---:|---:|
| SwiftF0 | **0.781** [0.768, 0.795] | 179.6 ± 6.7 | **16** | **0** | 1 |
| RMVPE | 0.768 [0.752, 0.783] | 13.6 ± 0.8 | **16** | **0** | 1 |
| FCPE | 0.728 [0.712, 0.742] | 27.8 ± 3.2 | 15 | 2 | 0 |
| TorchCREPE | 0.691 [0.673, 0.706] | 0.4 ± 0.0 | 12 | 3 | 2 |
| CREPE | 0.689 [0.672, 0.704] | 0.4 ± 0.0 | 12 | 3 | 2 |
| PESTO | 0.680 [0.663, 0.697] | 20.0 ± 0.1 | 12 | 3 | 2 |
| SHS | 0.657 [0.640, 0.672] | 1080.3 ± 9.1 | 10 | 6 | 1 |
| Praat | 0.651 [0.634, 0.667] | 525.3 ± 17.9 | 8 | 6 | 3 |
| RAPT | 0.640 [0.621, 0.655] | **1276.6** ± 14.8 | 8 | 7 | 2 |
| HarmoF0 | 0.639 [0.622, 0.653] | 4.1 ± 0.1 | 8 | 7 | 2 |
| SWIPE | 0.610 [0.589, 0.624] | 50.3 ± 0.2 | 5 | 10 | 2 |
| SPICE | 0.602 [0.582, 0.619] | 86.2 ± 2.7 | 5 | 10 | 2 |
| Harvest | 0.600 [0.584, 0.614] | 18.4 ± 0.4 | 5 | 10 | 2 |
| YAAPT | 0.560 [0.539, 0.580] | 50.5 ± 0.4 | 1 | 13 | 3 |
| PENN | 0.560 [0.539, 0.579] | 4.7 ± 0.1 | 1 | 13 | 3 |
| DIO | 0.560 [0.541, 0.577] | 211.9 ± 11.5 | 1 | 13 | 3 |
| BasicPitch | 0.557 [0.539, 0.573] | 59.7 ± 0.7 | 1 | 13 | 3 |
| pYIN | 0.506 [0.482, 0.525] | 9.6 ± 0.6 | 0 | 17 | 0 |
| REAPER (crashed)[^reaper] | - | - | - | - | - |

[^reaper]: REAPER has incomplete accuracy results and is unranked. Its speed is measured separately.

↑ Higher is better. ↓ Lower is better. Bold marks the best displayed value in each metric
column, including ties. SwiftF0 has the highest score. Its difference to RMVPE is not
statistically resolved.

Every tracker's output is resampled onto one 16 ms frame grid, then scored frame by frame:

```
cents_error = 1200 * log2(tracker pitch / reference pitch)
hit         = tracker voiced AND reference voiced AND |cents_error| < 50
precision   = Pr(hit | tracker voiced)
recall      = Pr(hit | reference voiced)
score       = F1, the harmonic mean of the two
```

The score averages pitch F1 equally over the eight recording conditions and ten corpora.
Clean audio is excluded.

Brackets show 95% confidence intervals. Beats and loses to count statistically significant
wins and losses after adjusting for all pairwise comparisons. Undetermined means the data
do not resolve the difference.

Speed is audio duration divided by the median CPU time over five rounds, with every tracker
pinned to one CPU core and the thread limits set to 1. A value of 20 means 20 seconds of audio
processed per second of CPU time. The value after ± is the standard deviation over the rounds.
See [Speed](BENCHMARK.md#speed) for the method and the measured CPU.

## Install

Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync --all-extras
```

To benchmark only some of the trackers, install just their extras:

```bash
uv sync --extra crepe --extra praat  # add trackers one --extra at a time
```

## Run

Download the dataset from [Hugging Face](https://huggingface.co/datasets/lars1234/pitch-benchmark),
or rebuild it from the raw corpora with [prepare/](prepare/README.md):

> [!NOTE]
> The dataset is licensed CC BY-NC-SA 4.0, for research use only.


```bash
uvx --from huggingface_hub hf download lars1234/pitch-benchmark eval.tar SHA256SUMS --repo-type dataset --local-dir dataset
(cd dataset && sha256sum -c --ignore-missing SHA256SUMS)
tar -xf dataset/eval.tar -C dataset
```

`train.tar` is only needed for training and is not downloaded by this command.

Point `--dataset` at the extracted `eval/` directory:

```bash
nohup uv run python run.py --dataset dataset/eval --out cells --workers 8 \
      > run.log 2>&1 &                        # the matrix: 9 panels x 10 corpora x 19 trackers
tail -f run.log                               # the last line reads "=== done" when the matrix is complete

uv run python speed.py --out cells            # speed, afterwards, on an otherwise idle machine
uv run python report.py --cells cells --out BENCHMARK.md    # rewrites the tables inside BENCHMARK.md
```

`--algorithms SwiftF0 Praat` restricts a run to the trackers named, for example those whose
extras are installed. `--datasets` and `--panels` restrict it the same way. Runs resume: a cell
whose file exists is skipped, so delete `cells/` to start over. The leaderboard in this README is
copied from the report.

Each tracker, corpus and condition runs in a separate process with the thread-limit environment
variables set to one. Set `--workers` to the number of physical CPU cores. The 19 trackers took
around 8 hours at `--workers 8` on an 8-core Ryzen 9 8945HS. `nohup` keeps the run active after
the terminal closes. Run the speed measurements afterwards on an otherwise idle machine.

## Changes

[CHANGELOG.md](CHANGELOG.md) lists the changes of every version.
[Version 1](https://github.com/lars76/pitch-benchmark/tree/v1) remains available under the v1 tag.

## Contributing

To add a tracker, open a pull request with a wrapper in `algorithms/` and its extra in
`pyproject.toml`, or open an issue asking for it and it will be benchmarked here.

## License

MIT, see LICENSE.
