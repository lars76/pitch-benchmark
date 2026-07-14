# Pitch Detection Benchmark

A comprehensive benchmark suite evaluating pitch detection algorithms across datasets covering speech, music, synthetic, and real-world audio conditions.

## Which Algorithm Should I Use?

**TL;DR Recommendations:**
- **Best Overall**: **SwiftF0** (90.2% accuracy, 90× faster than CREPE)
- **Need Maximum Speed**: **Praat** (2.8ms per second of audio, 84.7% accuracy)
- **Best Pitch Accuracy**: **CREPE** (85.3% accuracy, best RPA/RCA but slow and not good on all metrics)
- **Best Human singing**: **RMVPE** (87.2% accuracy, best on Vocadito and MIR-1K)

## Overall Results

The table below shows the harmonic-mean accuracy score for each algorithm across the seven benchmark datasets. The average score determines the overall ranking.

| **Algorithm** | **Bach10Synth** | **MDBStemSynth** | **MIR1K** | **NSynth** | **PTDB** | **SpeechSynth** | **Vocadito** | **Average** |
|---|---|---|---|---|---|---|---|---|
| **SwiftF0** | 97.5% | 92.0% | 95.0% | **89.3%** | 90.4% | **90.7%** | 92.6% | **90.2%** |
| RMVPE | 98.1% | 90.6% | **96.0%** | 68.2% | 88.9% | 90.6% | **96.4%** | 87.2% |
| CREPE | **98.5%** | 90.5% | 95.7% | 80.2% | 79.7% | 88.3% | 95.6% | 85.3% |
| PENN | 97.3% | **94.0%** | 89.0% | 63.3% | **91.0%** | 84.8% | 82.4% | 84.8% |
| Praat | 96.0% | 90.7% | 92.6% | 70.7% | 86.2% | 88.2% | 88.2% | 84.7% |
| SPICE | 95.0% | 89.4% | 92.7% | 68.8% | 77.8% | 87.9% | 92.3% | 82.5% |
| TorchCREPE | 96.7% | 85.1% | 71.4% | 83.8% | 78.3% | 79.7% | 89.0% | 80.6% |
| pYIN | 97.5% | 90.3% | 91.2% | 74.3% | 72.1% | 81.4% | 79.5% | 78.7% |
| RAPT | 91.9% | 79.6% | 82.4% | 54.6% | 68.4% | 74.3% | 87.5% | 73.5% |
| SWIPE | 77.8% | 65.6% | 77.1% | 51.4% | 66.6% | 77.1% | 66.6% | 65.9% |
| YAAPT | 58.5% | 39.6% | 82.0% | 6.4% | 69.8% | 83.5% | 88.6% | 60.0% |
| BasicPitch | 23.7% | 12.4% | 36.5% | 77.7% | 23.1% | 61.2% | 17.8% | 33.1% |

For a detailed breakdown of results, see [Benchmark Report](benchmark_report.md).

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

`pitch_benchmark.py --device {auto,cpu,cuda,mps}` selects the compute device for the neural trackers
(TorchCREPE, RMVPE, PENN); the DSP/own-runtime trackers always run on CPU. `auto` (the default) picks
`cuda → mps → cpu`. **cpu/cuda are the reproducible reference** for the leaderboard (`run.sh` pins
`--device cpu`, or run `DEVICE=cuda ./run.sh` on a CUDA box); **mps is a local speed option** (Apple GPU)
whose numerics differ slightly (within the 50-cent RPA tolerance). `speed_benchmark.py` defaults to CPU
only; pass `--devices cpu mps` (or `cuda`) to also time the GPU-capable trackers.

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

- [CHiME-Home](https://archive.org/details/chime-home) - Domestic background noise ([Foster et al., WASPAA 2015](https://ieeexplore.ieee.org/document/7314880)); pass with `--chime-dir`
- [DEMAND](https://zenodo.org/records/1227121) - Multi-environment acoustic noise ([Thiemann, Ito & Vincent, Proc. Mtgs. Acoust. 2013](https://hal.science/hal-00796707)); pass with `--demand-dir`

Organize datasets in a directory structure like:
```
datasets/
├── PTDB/
├── NSynth/
├── MDBStemSynth/
├── MIR1K/
├── Vocadito/
├── Bach10Synth/
├── MOCHA/              # raw CSTR per-speaker: <spk>_<num>.wav + .lar (--dataset MOCHA)
├── cmu_arctic_egg/            # extracted -WAVEGG: cmu_us_<spk>_arctic/orig/*.wav (--dataset CMUArctic)
├── avid/extracted/           # AVID/Repository 1/Spk*_*.wav stereo (--dataset AVID)
├── osf_glottis/bids_dataset/ # BIDS sub-XX/beh/*_physio.tsv.gz (--dataset OSFGlottis)
├── svd/healthy/              # extracted healthy.zip: <rec>/sentences/*.nsp + overview.csv (--dataset SVD)
├── aplawd/APLAWDW/           # extracted aplawdw.zip: <...>/*.wav + *.egg (--dataset APLAWD)
├── KEELE/KEELE/        # Bechtold jbof redistribution: <stem>/signal.wav + laryngograph.wav + pitch.npy (--dataset KEELE)
├── FDA/                      # raw .sig/.fx/.lar (--dataset FDA)
├── URMP/Dataset/    # multi-instrument music (--dataset URMP)
├── chime_home/      # noise source for --degradation chime
└── DEMAND/          # noise source for --degradation demand
```

### Usage

**1. Visualize Algorithms on Your Audio**
```bash
python visualize_algorithms.py your_audio.wav --selected_algorithms SwiftF0 CREPE Praat
```

**2. Run the whole benchmark** (clean leaderboard + robustness probe + speed + report)

Edit the dataset paths at the top of `run.sh`, then:
```bash
./run.sh
```
It is resumable: finished results are skipped, so re-running after adding an algorithm is cheap.

**3. Individual runs**

Clean is the default condition (no `--chime-dir` needed):
```bash
uv run python pitch_benchmark.py --dataset Vocadito --data-dir datasets/vocadito
```
Robustness to a degradation on a small probe (capped + truncated sample):
```bash
uv run python pitch_benchmark.py --dataset Vocadito --data-dir datasets/vocadito \
  --degradation pink --max-samples 30 --max-seconds 10
```
Degradations: `clean, white, pink, chime, demand, telephone, reverb, room` (`chime` needs `--chime-dir`, `demand` needs `--demand-dir`).

**4. Speed benchmark**
```bash
uv run python speed_benchmark.py
```

**5. Generate the report**
```bash
uv run python generate_report.py --results-dir results/ --output benchmark_report.md
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
timestamps are **measured** against synthetic signals with analytically known f0
(`tests/test_time_calibration.py`), and the same correction policy is applied to all trackers.
See [TIMING.md](TIMING.md) for the contract, the policy, and the per-tracker calibration table.

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
