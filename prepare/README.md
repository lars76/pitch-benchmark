# Recreating the dataset

This folder provides the code to recreate the pitch-benchmark dataset. The prepared
dataset is on [Hugging Face](https://huggingface.co/datasets/lars1234/pitch-benchmark);
use this code to rebuild it or to change how it is built.

## Quickstart

From this directory, download every dataset listed below into `raw/`, then:

```bash
uv sync
for name in PTDB MOCHA CMUArctic KEELE FDA APLAWD AVID OSFGlottis SVD; do
    uv run python -m scripts.build_consensus_labels --dataset "$name" --data-dir "raw/$name"
done
uv run python build.py
```

## Datasets

Use the exact directory names below under `raw/`.

### Eval corpora (scored)

| Dataset | What it is | Download | Extract to | Contains | Size | Files |
| --- | --- | --- | --- | --- | --- | --- |
| KEELE | 10 speakers, North Wind passage, laryngograph | [Zenodo](https://zenodo.org/records/3921794): `KEELE.zip`, keep the inner `KEELE/` | `raw/KEELE/` | `f1nw0000/ f2nw0000/ …` | 26 MB | 72 |
| FDA (Bagshaw) | 50 sentences x 2 speakers, studio + laryngograph | [`fda_eval.tar.gz`](https://www.cstr.ed.ac.uk/research/projects/fda/fda_eval.tar.gz) | `raw/FDA/` | `rl/ sb/ man/ src/` | 26 MB | 310 |
| APLAWD | 151 utterances x 10 British-RP speakers, speech + laryngograph | [`aplawdw.zip`](https://www.commsp.ee.ic.ac.uk/~sap/uploads/data/aplawdw.zip) | `raw/APLAWD/` | `c/ d/ l/ s/ w/ x/ doc/ matlab/` | 734 MB | 33,234 |
| AVID | 50 speakers, calibrated ~15 min sessions, speech + EGG | [Zenodo](https://zenodo.org/records/10524873): from `AVID.zip` extract `AVID/Repository 1/` | `raw/AVID/` | `Repository 1/` | 11 GB | 51 |
| OSF Glottis | Harvard sentences, 25-47 min sessions, speech + EGG + intraoral pressure | [`bids_dataset.zip`](https://osf.io/download/ntgak/) from [osf.io/5yn2f](https://osf.io/5yn2f/). Keep the inner `bids_dataset/` contents | `raw/OSFGlottis/` | `sub-06/ … sub-19/ code/ participants.tsv` | 2.9 GB | 61 |
| Saarbruecken Voice Database | German connected-speech phrases + EGG, healthy controls only | [`healthy.zip`](https://zenodo.org/records/16874898/files/healthy.zip). The loader uses 634 recordings with paired speech and EGG phrase files | `raw/SVD/` | `1/ 10/ 100/ …` | 6.4 GB | 19,281 |
| Vocadito | 40 solo singing clips | [Zenodo](https://zenodo.org/records/5578807) | `raw/Vocadito/` | `Audio/ Annotations/ vocadito_metadata.csv` | 72 MB | 203 |
| URMP | chamber ensembles, manually corrected per-track f0 | [`Eredis02/URMP`](https://huggingface.co/datasets/Eredis02/URMP): keep the `AuSep_*.wav`, `F0s_*.txt` and `Notes_*.txt` files | `raw/URMP/` | `01_Jupiter_vn_vc/ …` | 2.2 GB | 447 |
| Bach10-mf0-synth | resynthesized Bach chorales, exact f0 | [`Bach10-mf0-syth.tar.gz`](https://zenodo.org/records/1481156/files/Bach10-mf0-syth.tar.gz) | `raw/Bach10Synth/` | `audio_stems/ annotation_stems/ audio_mix/ annotation_mf0/` | 151 MB | 102 |
| SpeechSynth | synthetic Mandarin, exact f0 by construction | [`lightspeech_new.pt`](https://github.com/lars76/fastspeech2-clean/releases/download/models/lightspeech_new.pt) (LightSpeech, `d_model=512`), renamed | `raw/SpeechSynth/` | `speechsynth.pt` | 25 MB | 1 |

### Train corpora (never scored)

| Dataset | What it is | Download | Extract to | Contains | Size | Files |
| --- | --- | --- | --- | --- | --- | --- |
| MDB-stem-synth | resynthesized MedleyDB stems, exact f0 | [Zenodo](https://zenodo.org/records/1481172) | `raw/MDBStemSynth/` | `audio_stems/ annotation_stems/` | 5.0 GB | 462 |
| NSynth | acoustic instrument notes, 4 s each, labelled by nominal MIDI pitch | [Magenta](https://magenta.tensorflow.org/datasets/nsynth): the **train** split | `raw/NSynth/` | `audio/ examples.json` | 35 GB | 289,206 |
| PTDB-TUG | read speech + laryngograph, 20 speakers | [TU Graz](https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html): the `SPEECH DATA` archive | `raw/PTDB/` | `FEMALE/ MALE/` | 6.3 GB | 14,154 |
| MOCHA-TIMIT | 8 speakers x 460 TIMIT sentences + laryngograph | [CSTR](http://data.cstr.ed.ac.uk/mocha/): `fsew0`, `msak0`, `maps0` plus `faet0`, `falh0`, `ffes0`, `fjmw0`, `mjjn0` from `unchecked/`. Extract all into one flat directory | `raw/MOCHA/` | `faet0_001.wav faet0_001.lar …` | 1.9 GB | 22,921 |
| CMU Arctic | read speech + EGG channel | [festvox `orig/`](http://www.festvox.org/cmu_arctic/cmu_arctic/orig/): the `-WAVEGG` builds of `bdl`, `jmk`, `slt`, each keeping its `orig/` directory | `raw/CMUArctic/` | `cmu_us_bdl_arctic/ …` | 1.5 GB | 10,178 |

### Degradation sources (unlabelled)

| Dataset | Used by | What it is | Download | Extract to | Contains | Size | Files |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DEMAND | eval | 17 real ambiences, multichannel | [Zenodo](https://zenodo.org/records/1227121): the **16 kHz** archives | `raw/DEMAND/` | `DKITCHEN/ DLIVING/ …` | 2.4 GB | 272 |
| AISHELL-3 | eval | Mandarin speakers, mixed in as babble | [OpenSLR 93](https://www.openslr.org/93/): `data_aishell3.tgz` | `raw/AISHELL3/` | `train/ test/ spk-info.txt …` | 25 GB | 88,042 |
| MIR-1K | eval | karaoke accompaniment, left channel | [`MIR-1K.zip`](http://mirlab.org/dataset/public/MIR-1K.zip) | `raw/MIR1K/` | `Wavfile/ …` | 1.1 GB | 5,221 |
| OpenAIR | eval | measured room impulse responses | [webfiles.york.ac.uk/OPENAIR/IRs/](https://webfiles.york.ac.uk/OPENAIR/IRs/): mirror the `.wav` files under each of the 59 environment directories | `raw/OPENAIR/` | `1st-baptist-nashville/ …` | 4.7 GB | 773 |
| RIRS_NOISES | both | simulated and measured impulse responses | [OpenSLR 28](https://www.openslr.org/28/): `rirs_noises.zip` | `raw/RIRS/` | `simulated_rirs/ real_rirs_isotropic_noises/ …` | 3.5 GB | 61,273 |
| LibriSpeech | train | read English speech, babble | [OpenSLR 12](https://www.openslr.org/12/): `train-clean-100.tar.gz` | `raw/LIBRISPEECH/` | `103/ 1034/ …` | 6.2 GB | 29,124 |
| TAU2019 | train | urban acoustic scenes | [`hzhongresearch/tau2019`](https://huggingface.co/datasets/hzhongresearch/tau2019): already 16 kHz mono in scene directories | `raw/TAU2019/` | `airport/ bus/ …` | 4.3 GB | 14,402 |
| MUSDB18 | train | music stems | [`musdb18.zip`](https://zenodo.org/records/1117372/files/musdb18.zip) | `raw/MUSDB18/` | `train/ test/` | 5.3 GB | 152 |

Training and evaluation use different RIRS_NOISES directories: `simulated_rirs/` for training
and `real_rirs_isotropic_noises/` for evaluation. All other source datasets are separate.

The build excludes 46 MUSDB18 tracks that overlap with MDB-stem-synth, leaving 104 tracks
for background music. This prevents a recording from serving as both a training target and
background music.
