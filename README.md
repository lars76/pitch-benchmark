# Pitch Detection Benchmark

A comprehensive benchmark suite evaluating pitch detection algorithms across 8 datasets covering speech, music, synthetic, and real-world audio conditions.

## Which Algorithm Should I Use?

**TL;DR Recommendations:**
- **Best Overall**: **SwiftF0** (90.2% accuracy, 90× faster than CREPE)
- **Need Maximum Speed**: **Praat** (2.8ms per second of audio, 84.7% accuracy)
- **Best Pitch Accuracy**: **CREPE** (85.3% accuracy, best RPA/RCA but slow and not good on all metrics)
- **Best Human singing**: **RMVPE** (87.2% accuracy, best on Vocadito and MIR-1K)

## Overall Results

The table below shows the harmonic-mean accuracy score for each algorithm across the eight benchmark datasets. The average score determines the overall ranking.

| **Algorithm** | **Bach10Synth** | **MDBStemSynth** | **MIR1K** | **NSynth** | **PTDB** | **PTDBNoisy** | **SpeechSynth** | **Vocadito** | **Average** |
|---|---|---|---|---|---|---|---|---|---|
| **SwiftF0** | 97.5% | 92.0% | 95.0% | **89.3%** | 90.4% | 74.0% | **90.7%** | 92.6% | **90.2%** |
| RMVPE | 98.1% | 90.6% | **96.0%** | 68.2% | 88.9% | 68.5% | 90.6% | **96.4%** | 87.2% |
| CREPE | **98.5%** | 90.5% | 95.7% | 80.2% | 79.7% | 53.8% | 88.3% | 95.6% | 85.3% |
| PENN | 97.3% | **94.0%** | 89.0% | 63.3% | **91.0%** | **76.4%** | 84.8% | 82.4% | 84.8% |
| Praat | 96.0% | 90.7% | 92.6% | 70.7% | 86.2% | 65.3% | 88.2% | 88.2% | 84.7% |
| SPICE | 95.0% | 89.4% | 92.7% | 68.8% | 77.8% | 55.9% | 87.9% | 92.3% | 82.5% |
| TorchCREPE | 96.7% | 85.1% | 71.4% | 83.8% | 78.3% | 61.2% | 79.7% | 89.0% | 80.6% |
| pYIN | 97.5% | 90.3% | 91.2% | 74.3% | 72.1% | 43.2% | 81.4% | 79.5% | 78.7% |
| RAPT | 91.9% | 79.6% | 82.4% | 54.6% | 68.4% | 48.9% | 74.3% | 87.5% | 73.5% |
| SWIPE | 77.8% | 65.6% | 77.1% | 51.4% | 66.6% | 45.0% | 77.1% | 66.6% | 65.9% |
| YAAPT | 58.5% | 39.6% | 82.0% | 6.4% | 69.8% | 51.7% | 83.5% | 88.6% | 60.0% |
| BasicPitch | 23.7% | 12.4% | 36.5% | 77.7% | 23.1% | 12.6% | 61.2% | 17.8% | 33.1% |

For a detailed breakdown of results, see [Benchmark Report](benchmark_report.md).

## Running Your Own Benchmarks

### Installation

This project uses [uv](https://docs.astral.sh/uv/pip/environments/) (a fast Python package manager) for dependency management, but `conda` or `pip` will also work.

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
```

### Dataset Setup

Download the required datasets:

- [PTDB-TUG](https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html) - Speech with laryngograph ground truth
- [NSynth](https://magenta.tensorflow.org/datasets/nsynth) - Synthetic musical instruments
- [MDB-stem-synth](https://zenodo.org/records/1481172) - Synthetic music stems
- [MIR-1K](https://zenodo.org/records/3532216) - Vocal excerpts
- [Vocadito](https://zenodo.org/records/5578807) - Solo vocal recordings
- [Bach10-mf0-synth](https://zenodo.org/records/1481156/files/Bach10-mf0-syth.tar.gz) - Synthetic Bach compositions
- [CHiME-Home](https://archive.org/details/chime-home) - Background noise for testing

Organize datasets in a directory structure like:
```
datasets/
├── PTDB/
├── NSynth/
├── MDBStemSynth/
├── MIR1K/
├── Vocadito/
├── Bach10Synth/
└── chime_home/
```

### Usage

**1. Visualize Algorithms on Your Audio**
```bash
python visualize_algorithms.py your_audio.wav --selected_algorithms SwiftF0 CREPE Praat
```

**2. Speed Benchmark**
```bash
python speed_benchmark.py --signal-length 1.0 --n-runs 20
```

**3. Pitch Benchmark**

```bash
for dataset in PTDB NSynth MIR1K Vocadito MDBStemSynth Bach10Synth; do
  python pitch_benchmark.py \
    --dataset $dataset \
    --data-dir datasets/$dataset \
    --chime-dir datasets/chime_home
done
python pitch_benchmark.py --dataset PTDBNoisy --data-dir datasets/PTDB --chime-dir datasets/chime_home
python pitch_benchmark.py --dataset SpeechSynth --data-dir datasets/speechsynth.pt --chime-dir audio_datasets/chime_home
```

**4. Generate Report**

```bash
python generate_report.py --results-dir results/ --output benchmark_report.md
```

### Algorithm Implementations

The benchmark includes implementations of these algorithms:

**Neural Networks:**
- [SwiftF0](https://github.com/lars76/swift-f0) - Fast CNN-based pitch detection
- [CREPE](https://github.com/marl/crepe) - CNN-based pitch estimation
- [TorchCREPE](https://github.com/maxrmorrison/torchcrepe) - PyTorch CREPE implementation
- [PENN](https://github.com/interactiveaudiolab/penn) - Pitch-Estimating Neural Networks
- [BasicPitch](https://github.com/spotify/basic-pitch) - Spotify's multi-instrument pitch detector
- [SPICE](https://www.tensorflow.org/hub/tutorials/spice) - Self-supervised pitch estimation
- [RMVPE](https://github.com/yxlllc/RMVPE) - A Robust Model for Vocal Pitch Estimation in Polyphonic Music

**Classical Methods:**
- [Praat](https://github.com/YannickJadoul/Parselmouth) - Autocorrelation-based
- [pYIN](https://librosa.org/doc/main/generated/librosa.pyin.html) - Probabilistic YIN
- [YAAPT](https://bjbschmitt.github.io/AMFM_decompy/pYAAPT.html) - Yet Another Algorithm for Pitch Tracking
- [RAPT](https://pysptk.readthedocs.io/en/latest/generated/pysptk.sptk.rapt.html) - Robust Algorithm for Pitch Tracking
- [SWIPE](https://pysptk.readthedocs.io/en/latest/generated/pysptk.sptk.swipe.html) - Sawtooth Waveform Inspired Pitch Estimator
- [DIO](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) - Distributed Inline-filter Operation: multi-band zero-crossing F0 estimator from the WORLD vocoder (via pyworld), with StoneMask refinement
- [Harvest](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) - High-performance F0 estimator using band-pass filter candidates refined with instantaneous frequency and contour smoothing (via pyworld)

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
