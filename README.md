# Pitch Detection Benchmark

A comprehensive benchmark suite evaluating pitch detection algorithms across datasets covering speech, music, synthetic, and real-world audio conditions.

## Which Algorithm Should I Use?

> **v2 rebuild in progress.** The scoring system was rewritten (see
> [Benchmark Report](benchmark_report.md) for the metric definitions); every algorithm
> is being re-measured under it. The table below lists only trackers with a complete
> v2 run — more are added as they finish, so treat it as a growing leaderboard, not a
> final ranking.

**From the trackers measured so far:**
- **Best overall**: **Praat** — no weak axis, and the fastest of the two.
- **Best on real recordings**: **RMVPE** — leads on clean accuracy and is by far the
  most noise-robust, but it collapses on synthetic signal classes (pure tones,
  missing-fundamental, IRN) that it never saw in training. If your audio is real
  speech or music, that weakness may not reach you; if you feed it synthesised or
  degenerate signals, it will.

That contrast is the point of the multi-track design: one number cannot tell you
whether a tracker is strong everywhere or excellent in one place and broken in
another. Read the track columns, not just the overall.

## Overall Results

The overall score is the harmonic mean of the six track scores (see [METRICS.md](METRICS.md) for definitions).

Reference cells are being regenerated against the current metric definitions; this table is rendered from them by `generate_report.py --readme README.md`.

## Running Your Own Benchmarks

### Installation

This project uses [uv](https://docs.astral.sh/uv/). The torch backend is selected per platform
automatically (CUDA 12.6 on Linux, CPU/MPS on macOS); TensorFlow runs on CPU. Requires Python 3.10.

Each algorithm backend is an optional extra, so you install only what you want to run:

```bash
uv sync                              # core: all datasets + the pYIN and RMVPE trackers
uv sync --extra dio --extra swipe    # a few algorithms (one --extra per backend)
uv sync --all-extras                 # the full benchmark (every algorithm)
```

The extra name matches the algorithm; `pYIN` and `RMVPE` are in the core and need no extra:

| extra | algorithm | extra | algorithm |
|---|---|---|---|
| `crepe` | CREPE | `swiftf0` | SwiftF0 |
| `spice` | SPICE | `dio` | DIO |
| `torchcrepe` | TorchCREPE | `harvest` | Harvest |
| `penn` | PENN | `rapt` | RAPT |
| `praat` | Praat | `swipe` | SWIPE |
| `yaapt` | YAAPT | `basicpitch` | BasicPitch |
| `reaper` | REAPER | | |

Prefix commands with `uv run` to use the managed environment (e.g. `uv run python ...`).

### Device (CPU / CUDA / MPS)

`evaluate.py --device {auto,cpu,cuda,mps}` selects the compute device for the neural trackers
(TorchCREPE, RMVPE, PENN); the DSP/own-runtime trackers always run on CPU. **cpu/cuda are the
reproducible reference** for the leaderboard (the default is cpu); **mps is a local speed option**
(Apple GPU) whose numerics differ slightly (within the 50-cent RPA tolerance). The speed track
always times cpu and additionally the requested GPU device.

### Dataset Setup

**Pitch-annotated datasets** (the clean leaderboard):

- [PTDB-TUG](https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html) - Speech with laryngograph ground truth ([Pirker et al., Interspeech 2011](https://www.isca-archive.org/interspeech_2011/pirker11_interspeech.html)); v2 includes all 4718 utterances (no file exclusion list), with per-frame label quality handled by the consensus labels
- [NSynth](https://magenta.tensorflow.org/datasets/nsynth) - Synthetic musical instruments ([Engel et al., 2017](https://arxiv.org/abs/1704.01279))
- [MDB-stem-synth](https://zenodo.org/records/1481172) - Resynthesized MedleyDB stems with exact f0 ([Salamon et al., ISMIR 2017](http://synthdatasets.weebly.com/mdb-stem-synth.html))
- [MIR-1K](https://zenodo.org/records/3532216) - Vocal excerpts ([Hsu & Jang, IEEE TASLP 2010](https://ieeexplore.ieee.org/document/5153305))
- [Vocadito](https://zenodo.org/records/5578807) - Solo vocal recordings ([Bittner et al., 2021](https://arxiv.org/abs/2110.05580))
- [Bach10-mf0-synth](https://zenodo.org/records/1481156/files/Bach10-mf0-syth.tar.gz) - Resynthesized Bach10 with exact f0 ([Duan et al., IEEE TASLP 2010](https://ieeexplore.ieee.org/document/5445037); analysis/synthesis by [Salamon et al., ISMIR 2017](http://synthdatasets.weebly.com/mdb-stem-synth.html))

(SpeechSynth needs no download; it is rendered at runtime from a LightSpeech TTS checkpoint.)

**EGG (laryngograph) speech corpora.** These are recorded with an electroglottograph, so the f0 is
derived from the glottal (laryngograph) channel rather than annotated by ear. **To use them you only
(1) download + extract the original dataset and (2) run the benchmark**. The loader decodes audio
straight from the raw extracted download (each parses that corpus's native format), and the ground
truth ships in the repo. There is no preprocessing step to run.

Ground truth comes from one of two sources:
- `consensus` (default): cross-family EGG labels (Praat / differentiated-EGG / Harvest), **committed
  in the repo** as `datasets/laryngograph/<NAME>.npz`.
- `reference`: the dataset authors' own shipped f0, read from the archive, available only where the
  download ships one: PTDB (`REF/.f0`), KEELE (its reference track), FDA (`.fx`). The other corpora
  ship audio + EGG but no f0, so they are consensus-only.

(SVD: extract `healthy.zip` only, the healthy-control subset; SVD is otherwise a voice-disorder
database whose pathological voices make unreliable f0 ground truth. MOCHA's CSTR download is raw
`.wav` + `.lar`; KEELE is distributed pre-packaged as `signal.wav` + `laryngograph.wav`; the loader
reads whichever form is present for each.)

> **Maintainers only** (regenerating the committed labels; end users never run this): `uv sync
> --extra praat --extra harvest` then `scripts/build_consensus_labels.py --dataset <NAME> --data-dir
> <extracted-root>` for every EGG corpus, and commit the resulting npz.

- [MOCHA-TIMIT](https://www.cstr.ed.ac.uk/research/projects/artic/mocha.html) - 9 speakers x 460 TIMIT sentences, speech + laryngograph (`.wav` + `.lar`, 16 kHz, 1024-byte NIST/SPHERE headers). CSTR, free non-commercial; download per-speaker from [data.cstr.ed.ac.uk/mocha](http://data.cstr.ed.ac.uk/mocha/)
- [CMU Arctic](http://www.festvox.org/cmu_arctic/) - phonetically balanced English, EGG-recorded (festvox; free). Download the `-WAVEGG` stereo distribution (speech + EGG channel); the plain mono package has no EGG and cannot be used (there is no shipped f0 to fall back on)
- [KEELE](https://zenodo.org/records/3921794) - 10 speakers reading the North Wind passage, speech + laryngograph reference ([Plante et al., 1995](https://www.isca-archive.org/eurospeech_1995/plante95_eurospeech.html)). NOTE: the first-party Keele/SAM distribution is defunct (the KTH/MIT `lost-contact` mirror is offline and not archived in the Wayback Machine), so KEELE is the one EGG corpus obtained from a curated redistribution rather than a first-party raw download: Bechtold's Zenodo compilation "[Speech and Noise Corpora for Pitch Estimation of Human Speech](https://zenodo.org/records/3921794)" ([Bechtold, 2020, dissertation replication set](https://github.com/bastibe/Replication-Dataset-Scripts)). It repackages the corpus as a [jbof](https://github.com/bastibe/jbof) dataframe (per item `signal.wav` + `laryngograph.wav` + `pitch.npy`) while preserving the original 20 kHz / 16-bit audio unchanged, so the loader reads that faithful packaging directly.
- [FDA](https://www.cstr.ed.ac.uk/research/projects/fda/) - Bagshaw 50-sentence x 2-speaker f0 evaluation set with `.lar` laryngograph ([`fda_eval.tar.gz`](https://www.cstr.ed.ac.uk/pcb/fda_eval.tar.gz))
- [SVD](http://stimmdb.coli.uni-saarland.de/) - Saarbruecken Voice Database, speech + separate EGG (web export; helper: [svd-downloader](https://github.com/rijulg/svd-downloader))
- [AVID](https://zenodo.org/records/10524873) - Aalto Vocal Intensity Database: 50 speakers, calibrated speech + EGG at four intensity levels (open, Zenodo; [Alku et al., Speech Communication 2024](https://www.sciencedirect.com/science/article/pii/S0167639324000116))
- OSFGlottis - Alku "Speech and EGG Simultaneous Recordings" ([Kielipankki](https://research.aalto.fi/en/datasets/speech-and-egg-electroglottography-simultaneous-recordings/); CLARIN academic/non-commercial)
- APLAWD - 151 short-utterance types x 10 repetitions x 10 British-RP speakers, speech + laryngograph (EGG) at 20 kHz (recorded at UCL 1987-88 for the SPAR project; Lindsey, Breen & Nevard). We use the **APLAWDW** repackaging by M. Brookes, Imperial College London (2015): [`aplawdw.zip`](https://www.commsp.ee.ic.ac.uk/~sap/uploads/data/aplawdw.zip) from [Imperial SAP](https://www.commsp.ee.ic.ac.uk/~sap/resources/aplawdw/) (server occasionally 500s; the download URL and GCI reference markings are also documented at [serwy/aplawdw](https://github.com/serwy/aplawdw))

**Multi-instrument music.**

- [URMP](https://labsites.rochester.edu/air/projects/URMP.html) - 44 classical chamber pieces with manually corrected per-track f0 and note transcriptions ([Li et al., IEEE TMM 2019](https://ieeexplore.ieee.org/document/8411155); access via the linked form)

> **Disclosure.** Several EGG-speech corpora (PTDB, MOCHA, CMU Arctic, and the wider EGG pool) plus
> MDB-stem-synth are commonly used as *training* data for pitch trackers. Their leaderboard cells can
> therefore be home data for any model trained on them and should be read with that caveat; the
> resynthesized-music and held-out speech corpora (KEELE, FDA, ...) are the fair comparison points.

**Noise sources** (only for the real-noise robustness conditions `--degradation chime|demand`):

- [CHiME-Home](https://archive.org/details/chime-home) - Domestic background noise ([Foster et al., WASPAA 2015](https://ieeexplore.ieee.org/document/7314880)); pass as `chime=<dir>` (the recorded-noise corpus for the chime condition)
- [DEMAND](https://zenodo.org/records/1227121) - Multi-environment acoustic noise ([Thiemann, Ito & Vincent, Proc. Mtgs. Acoust. 2013](https://hal.science/hal-00796707)); pass as `demand=<dir>`

**Dataset locations**: you tell the benchmark where each dataset's files start, explicitly —
`--data "PTDB=/my/SPEECH DATA" KEELE=/my/KEELE/KEELE ...` (the loader reads its corpus's
documented structure from exactly that directory; nothing is searched for). There is one
format and no fallback: every dataset you evaluate needs an entry, except the corpora
bundled in this repo. A layout like the one below keeps the entries short, but the names are
yours to choose:

```
your-datasets/                       # each entry passed with --data NAME=DIR
├── PTDB/                            # PTDB raw (archive extracts as "SPEECH DATA", rename to PTDB)
├── KEELE/                           # Bechtold jbof: <stem>/signal.wav + laryngograph.wav + pitch.npy
│                                    #   (archive nests KEELE/KEELE, flatten one level)
├── FDA/                             # raw .sig/.fx/.lar
├── Vocadito/
├── Bach10Synth/                     # content of the archive's Bach10-mf0-synth/ folder
├── OSFGlottis/                      # BIDS bids_dataset/sub-XX/... inside
├── AVID/                            # extracted/AVID/Repository 1/Spk*_*.wav inside
├── MIR1K/                           # archive extracts as MIR-1K, rename
├── URMP/                            # multi-instrument music, Dataset/ inside
├── MDBStemSynth/                    # archive extracts as MDB-stem-synth, rename
├── SVD/                            # content of healthy.zip: <rec>/sentences/*.nsp + overview.csv
├── NSynth/                          # the nsynth-test held-out split, rename
├── APLAWD/                          # content of aplawdw.zip's APLAWDW/ folder: *.wav + *.egg
├── chime_home/                      # real-noise source for the chime degradation (upstream name)
├── DEMAND/                          # real-noise source for the demand degradation
├── MOCHA/                           # raw CSTR <spk>_<num>.wav + .lar
├── CMUArctic/                       # -WAVEGG cmu_us_<spk>_arctic/orig/*.wav (folder: cmu_arctic_egg if you keep the download name; point --data at it)
└── M4Singer/                        # score-grade pitch GT: voicing-reliable; off by default -- include only by passing its path
```
SpeechSynth needs no download, it renders at runtime from `datasets/speechsynth.pt` inside the repo;
being bundled it is the one dataset you can opt in without a path (`--datasets SpeechSynth`).

**Datasets are opt-in.** A dataset is in a run only when you name it — a path via `--data NAME=DIR`,
or an explicit `--datasets NAME` (a bundled dataset then needs no path). Nothing rides along
automatically and nothing is hard-excluded, so a corpus with score-grade pitch GT (M4Singer) is
in a run only if you pass its path — with the understanding that scoring it shifts `theta*` and
every pitch table.

### Usage

**1. Visualize Algorithms on Your Audio**
```bash
python visualize_algorithms.py your_audio.wav --selected_algorithms SwiftF0 CREPE Praat
```

**2. Run the benchmark.** `evaluate.py` is the ONE entry point for all four suites (frame
accuracy + robustness, note transcription, synthetic signals, speed) and the report. It writes
cached cells into `--output-dir` (default `results/`); every run is **opt-in** (a dataset or
algorithm runs only if you name it) and **resumable** (finished cells are skipped, so re-running
after adding an algorithm or dataset only computes the new cells).

The knobs, composed freely:

| flag | selects | default |
|---|---|---|
| `--data NAME=DIR …` | dataset + `chime`/`demand` corpus locations | none (name it to include it) |
| `--suites {frame,note,synthetic,speed}` | which suites | all four |
| `--algorithms NAME …` | which trackers | all installed |
| `--datasets NAME …` | narrow to a subset of the named datasets | all named |
| `--conditions NAME …` | narrow frame degradations (frame only) | all procedural |
| `--max-clips N --max-seconds S` | cap (sample) every dataset-backed cell (frame + note) | uncapped |
| `--device {cpu,mps,cuda}` · `--workers N` · `--report` | how / where | cpu · 4 · off |

**The affordable two-tier workflow.** A full-corpus sweep of all 13 conditions across many
trackers is not affordable, so run it in two tiers into one `results/`:

```bash
# TIER 1 -- the cheap probe (fast): capped frame (all conditions) + synthetic + speed
uv run python evaluate.py --data <paths> --suites frame synthetic speed \
  --max-clips 30 --max-seconds 10 --device mps --workers 6

# TIER 2 -- the verdict (slow, full corpus): the clean headline + full note, then the report
uv run python evaluate.py --data <paths> --suites frame note --conditions clean \
  --device mps --workers 6 --report
```

Tier 1's cap makes every degraded condition a 30-clip probe (and note a fast capped pass) — the
affordable leaderboard. Tier 2 adds the uncapped clean cell that `theta*` and Correctness read,
plus full-corpus note; the capped clean from Tier 1 stays as the Noise track's Δ-from-clean
partner. Capped and uncapped cells coexist under their own keys (the cap is part of a cell's
identity), and `theta*` / `track_notes` prefer the uncapped ones. Only the uncapped Tier 2 is
certifiable — `assert_full()` rejects any capped cell. Because everything is cached, you can slice
by cost however you like — fast DSP trackers today, slow neural ones (`--algorithms CREPE RMVPE
…`) tomorrow — and nothing recomputes.

**3. Narrowed / exploratory runs**, same entry point:
```bash
# one tracker, one dataset, frame only
uv run python evaluate.py --data "Vocadito=/data/vocadito" --algorithms Praat --suites frame
# one condition on one dataset
uv run python evaluate.py --data "KEELE=/data/KEELE/KEELE" --conditions pink --suites frame
# the dataless suites need no --data
uv run python evaluate.py --suites synthetic speed
```
Conditions (13): `clean, white, pink, chime, demand, telephone, codec, reverb, room, gain, fade,
pink_snr+10, pink_snr-5`. The recorded conditions chime/demand each need their corpus via
`--data chime=<dir> demand=<dir>`; passing the corpora adds those two conditions to the frame axis.

**4. Programmatic use** (your own tracker class, no registry edits):
```python
from evaluate import run_cells, compare
# datasets is ONE {name: path|None} map -- keys are the matrix, values are locations
# (None = the dataset's own default; only bundled corpora have one). Uncapped = verdict mode.
cells = run_cells([MyTracker, "SwiftF0"], datasets={"KEELE": "/data/KEELE/KEELE"})
d = compare(cells, "MyTracker", "SwiftF0", metric="voicing_f1")
print(d.value, d.lo, d.hi, d.significant)

# the six track scores, each with its interval where one is sound
from evaluate import track_scores
tracks = track_scores(cells, "MyTracker")
print(tracks.correctness)        # Score(value=..., lo=..., hi=...)
```

**5. Generate the report separately** (or just pass `--report` above):
```bash
uv run python generate_report.py --results results/ --out benchmark_report.md
```

### Algorithm Implementations

The benchmark includes implementations of these algorithms:

Each entry links its implementation, followed by the paper it is based on.

**Neural Networks:**
- [SwiftF0](https://github.com/lars76/swift-f0) - Fast CNN spectrogram pitch detector ([Nieradzik, 2025](https://arxiv.org/abs/2508.18440))
- [CREPE](https://github.com/marl/crepe) - CNN on the raw waveform ([Kim, Salamon, Li & Bello, ICASSP 2018](https://arxiv.org/abs/1802.06182))
- [TorchCREPE](https://github.com/maxrmorrison/torchcrepe) - PyTorch port of CREPE ([Kim et al., 2018](https://arxiv.org/abs/1802.06182))
- [PENN](https://github.com/interactiveaudiolab/penn) - Pitch-Estimating Neural Networks ([Morrison et al., ICASSP 2023](https://arxiv.org/abs/2301.12258))
- [BasicPitch](https://github.com/spotify/basic-pitch) - Spotify's instrument-agnostic transcription/multipitch model ([Bittner et al., ICASSP 2022](https://arxiv.org/abs/2203.09893))
- [SPICE](https://www.tensorflow.org/hub/tutorials/spice) - Self-supervised pitch estimation ([Gfeller et al., IEEE/ACM TASLP 2020](https://arxiv.org/abs/1910.11664))
- [RMVPE](https://github.com/yxlllc/RMVPE) - Robust Model for Vocal Pitch Estimation in polyphonic music ([Wei, Cao, Dan & Chen, Interspeech 2023](https://arxiv.org/abs/2306.15412))

**Classical / DSP Methods:**
- [Praat](https://github.com/YannickJadoul/Parselmouth) - Autocorrelation pitch tracker (Boersma, 1993), via Parselmouth ([Jadoul et al., 2018](https://doi.org/10.1016/j.wocn.2018.07.001))
- [REAPER](https://github.com/google/REAPER) - Robust Epoch And Pitch EstimatoR (D. Talkin, Google), via [pyreaper](https://github.com/r9y9/pyreaper)
- [pYIN](https://librosa.org/doc/main/generated/librosa.pyin.html) - Probabilistic YIN ([Mauch & Dixon, ICASSP 2014](https://ieeexplore.ieee.org/document/6853678))
- [YAAPT](https://bjbschmitt.github.io/AMFM_decompy/pYAAPT.html) - Yet Another Algorithm for Pitch Tracking ([Zahorian & Hu, JASA 2008](https://doi.org/10.1121/1.2967862))
- [RAPT](https://pysptk.readthedocs.io/en/latest/generated/pysptk.sptk.rapt.html) - Robust Algorithm for Pitch Tracking (Talkin, 1995), via pysptk
- [SWIPE](https://pysptk.readthedocs.io/en/latest/generated/pysptk.sptk.swipe.html) - Sawtooth Waveform Inspired Pitch Estimator ([Camacho & Harris, JASA 2008](https://doi.org/10.1121/1.2951592)), via pysptk
- [DIO](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) - Zero-crossing/candidate F0 with StoneMask refinement, from the WORLD vocoder ([Morise et al., IEICE 2016](https://doi.org/10.1587/transinf.2015EDP7457)), via pyworld
- [Harvest](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) - Band-pass candidate F0 refined by instantaneous frequency, from WORLD ([Morise, Interspeech 2017](https://www.isca-archive.org/interspeech_2017/morise17_interspeech.html)), via pyworld

### Frame timing fairness

Predictions and labels are compared frame by frame on one shared grid, so every wrapper's
timestamps are **measured** against synthetic signals with analytically known f0 -- a
triangle chirp whose alternating slope separates a constant frequency bias from a timestamp
offset -- and the same correction policy is applied to every tracker. Each wrapper documents
its own measured offset in its source.

## 🤝 Contributing

Contributions are welcome! To add a new algorithm, you can either submit a Pull Request with your own implementation or create an Issue to request it, and I will run the benchmark for you.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{nieradzik2025swiftf0,
      title={SwiftF0: Fast and Accurate Monophonic Pitch Detection},
      author={Lars Nieradzik},
      year={2025},
      eprint={2508.18440},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2508.18440},
}
```
