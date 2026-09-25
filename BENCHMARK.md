# Pitch benchmark

## The design

For each tracker, the headline score is the mean pitch F1 across $C$ corpora and eight
recording conditions:

$$
\mathrm{Score} = \frac{1}{8C}\sum_{c=1}^{C}\sum_{p=1}^{8} F_{c,p}(\theta),
\qquad F = \frac{2H}{P+V}.
$$

The counts include all eligible test frames in each corpus and condition:

- $H$: voiced predictions with a pitch error below 50 cents (half a semitone).
- $P$: frames predicted as voiced.
- $V$: voiced reference frames.

The pitch score excludes voiced reference frames whose pitch cannot be verified or falls
outside the corpus search range. Pitch precision $H/P$ is the fraction of voiced predictions
with the correct pitch. Pitch recall $H/V$ is the fraction of voiced reference frames detected
with the correct pitch. Pitch F1 is their harmonic mean and is zero when $P+V=0$.

Each corpus and condition has equal weight, so larger datasets do not contribute more to
the headline score.

Each tracker uses one confidence threshold $\theta$ for all test corpora and conditions.
The benchmark chooses this threshold on separate calibration clips. All trackers receive
identical rendered audio. Trackers without continuous confidence retain their own voicing
decision.

The paired bootstrap resamples calibration and test clips within each corpus and repeats
threshold selection. Each draw uses the same clips for all trackers and conditions. Its
intervals apply to the fixed corpora and speakers. Independent resampling approximates the
systematic window selection in [select.py](prepare/corpora/select.py).

The adapters align predictions to a shared 16 ms frame grid and extend the first and last
predictions to the clip edges. A failed clip counts as missed voiced frames. If an entire
run for one corpus and condition fails, the tracker is unranked.

The full factorial design measures the performance loss caused by three recording factors,
both separately and together:

| Factor | Effect on the audio |
|---|---|
| Scene | Added background sound, including noise, competing speech and music |
| Room | Reverberation from room impulse responses |
| Microphone | Filtering that changes the frequency response |

Each factor can be on or off, giving $2^3=8$ versions of every clip. Comparing these versions
shows how much each factor affects the score and whether its effect changes when another
factor is present. For example, microphone filtering may cause a larger loss in background
sound than in quiet audio. This is an interaction. These comparisons help users choose a
tracker for their recording conditions. Clean audio provides an additional reference.

V1 combines six metrics into one score and chooses thresholds on the same clips it scores.
V2 uses pitch F1 and chooses thresholds on separate calibration clips.

## Datasets

### Evaluation corpora

The speech corpora with electroglottography (EGG) use reference labels computed from their
laryngograph recordings by three estimators. At least two must detect voicing, and at least
two pitch estimates must agree for the reference pitch to be accepted. An energy gate on
the microphone signal can exclude quiet onsets and tails. The consensus code is in
[build_consensus_labels.py](prepare/scripts/build_consensus_labels.py).

- [KEELE](https://zenodo.org/records/3921794) - 10 speakers reading the North Wind passage with a laryngograph
- [FDA](https://www.cstr.ed.ac.uk/research/projects/fda/) - Bagshaw/CSTR, 50 sentences x 2 speakers, studio 20 kHz with a laryngograph
- [APLAWD](https://www.commsp.ee.ic.ac.uk/~sap/uploads/data/aplawdw.zip) - 151 utterances x 10 British-RP speakers, speech + laryngograph (Brookes's APLAWDW repackaging)
- [AVID](https://zenodo.org/records/10524873) - Aalto Vocal Intensity Database: 50 speakers, calibrated ~15 min recordings, speech + EGG
- [OSF Glottis](https://osf.io/5yn2f/) - Harvard sentences with EGG and intraoral pressure, sampled from 25–47 minute sessions
- [Saarbruecken Voice Database](https://zenodo.org/records/16874898) - German connected-speech phrases with EGG, healthy-control subset only
- SpeechSynth - synthetic Mandarin speech with a known pitch contour imposed by WORLD resynthesis. [speechsynth.py](prepare/corpora/speechsynth.py) generates it from the [LightSpeech checkpoint](https://github.com/lars76/fastspeech2-clean/releases/download/models/lightspeech_new.pt) of [fastspeech2-clean](https://github.com/lars76/fastspeech2-clean)
- [Vocadito](https://zenodo.org/records/5578807) - Solo vocal recordings ([Bittner et al., 2021](https://arxiv.org/abs/2110.05580))
- [URMP](https://huggingface.co/datasets/Eredis02/URMP) - Classical chamber pieces with manually corrected per-track f0 ([Li et al., IEEE TMM 2019](https://ieeexplore.ieee.org/document/8411155))
- [Bach10-mf0-synth](https://zenodo.org/records/1481156/files/Bach10-mf0-syth.tar.gz) - Resynthesized Bach10 with exact f0 ([Duan et al., IEEE TASLP 2010](https://ieeexplore.ieee.org/document/5445037))

Background scenes add one or two sources: synthetic colored noise,
[DEMAND](https://zenodo.org/records/1227121) ambience,
[AISHELL-3](https://www.openslr.org/93/) speech or
[MIR-1K](http://mirlab.org/dataset/public/MIR-1K.zip) accompaniment.
The target-to-background ratio is drawn from three ranges: −6 to 2 dB, 2 to 8 dB, and
8 to 20 dB. Room conditions use measured impulse responses from
[RIRS_NOISES](https://www.openslr.org/28/) and
[OpenAIR](https://webfiles.york.ac.uk/OPENAIR/IRs/).
Microphone conditions apply high-pass and low-pass filtering, with an upper cutoff
between 3.4 and 7.5 kHz. Each clip uses the same sampled sources and parameters across
the eight factor combinations. Only the enabled factors change. The rendering code is in
[stages.py](prepare/stages.py), with settings in [constants.py](prepare/constants.py).

Background gain is adjusted after room and microphone filtering to preserve the selected
target-to-background ratio. Diffuse noise and ambience are added after reverberation.
Point sources pass through the room response. Microphone filtering also delays different
frequencies by different amounts.

<!-- dataset-stats -->
### Sample size

| Corpus | Test clips | Calibration clips / groups | Mean clip length (s) |
|---|---:|---:|---:|
| APLAWD | 16 | 24 / 4 | 0.9 |
| AVID | 26 | 14 / 13 | 9.9 |
| Bach10Synth | 23 | 17 / 4 | 9.2 |
| FDA | 40 | 0 / 0 | 3.3 |
| KEELE | 20 | 13 / 4 | 9.7 |
| OSFGlottis | 25 | 15 / 5 | 10.0 |
| SVD | 26 | 14 / 14 | 1.9 |
| SpeechSynth | 26 | 14 / 14 | 1.9 |
| URMP | 25 | 15 / 11 | 9.9 |
| Vocadito | 24 | 16 / 8 | 9.7 |

Each source clip is counted once, before rendering the recording conditions. Calibration clips in `valid/` come from separate speakers or recording groups and are used only to select the confidence threshold. Corpora without calibration clips use the threshold selected on the other corpora.

### Pitch distribution

| Corpus | Voiced (%) | f0 p5 / p50 / p95 (Hz) | Search range (Hz) | Out of range (%) | f0 by band |
|---|---:|---:|---:|---:|---:|
| APLAWD | 63 | 106-162-293 | 65-400 | 0.2 | low 91%, mid 9% |
| AVID | 49 | 98-168-324 | 65-400 | 1.3 | bass 2%, low 85%, mid 13% |
| Bach10Synth | 93 | 111-296-519 | 65-1200 | 0.0 | low 39%, mid 61% |
| FDA | 42 | 91-203-297 | 65-400 | 0.0 | low 79%, mid 21% |
| KEELE | 58 | 80-181-319 | 65-400 | 1.1 | bass 5%, low 81%, mid 14% |
| OSFGlottis | 29 | 102-204-261 | 65-400 | 0.1 | low 93%, mid 6% |
| SVD | 74 | 87-177-273 | 65-400 | 0.5 | bass 2%, low 89%, mid 8% |
| SpeechSynth | 49 | 112-211-275 | 65-400 | 0.0 | low 90%, mid 10% |
| URMP | 80 | 110-332-887 | 33-1200 | 0.8 | low 28%, mid 60%, high 9%, vhigh 2% |
| Vocadito | 69 | 114-234-366 | 80-1000 | 0.3 | low 62%, mid 38% |

Pitch bands across voiced frames, with equal weight per corpus: bass 1.1%, low 73.8%, mid 23.9%, high 1.0%, vhigh 0.2%.

Statistics describe the test clips. p5, p50 and p95 are the 5th, 50th and 95th pitch percentiles. Out of range is the percentage of verified voiced reference frames outside the search range. These frames are excluded from pitch F1. Bands: bass <80 Hz, low 80–260, mid 260–650, high 650–1050, vhigh ≥1050 Hz. Bands under 1% are omitted.

Trackers use the search range to restrict pitch candidates where supported. Otherwise, predictions outside the range are clamped to its nearest boundary.

<!-- /dataset-stats -->

<!-- report -->
## Results

### Overall performance

| Tracker | Pitch F1@50c ↑ [95% CI] | Voicing F1 ↑ | Beats ↑ | Loses to ↓ | Undetermined |
|---|---:|---:|---:|---:|---:|
| SwiftF0 | **0.781** [0.768, 0.795] | 0.844 | **16** | **0** | 1 |
| RMVPE | 0.768 [0.752, 0.783] | 0.837 | **16** | **0** | 1 |
| FCPE | 0.728 [0.712, 0.742] | 0.812 | 15 | 2 | 0 |
| TorchCREPE | 0.691 [0.673, 0.706] | 0.761 | 12 | 3 | 2 |
| CREPE | 0.689 [0.672, 0.704] | 0.781 | 12 | 3 | 2 |
| PESTO | 0.680 [0.663, 0.697] | **0.850** | 12 | 3 | 2 |
| SHS | 0.657 [0.640, 0.672] | 0.773 | 10 | 6 | 1 |
| Praat | 0.651 [0.634, 0.667] | 0.815 | 8 | 6 | 3 |
| RAPT | 0.640 [0.621, 0.655] | 0.799 | 8 | 7 | 2 |
| HarmoF0 | 0.639 [0.622, 0.653] | 0.727 | 8 | 7 | 2 |
| SWIPE | 0.610 [0.589, 0.624] | 0.761 | 5 | 10 | 2 |
| SPICE | 0.602 [0.582, 0.619] | 0.751 | 5 | 10 | 2 |
| Harvest | 0.600 [0.584, 0.614] | 0.782 | 5 | 10 | 2 |
| YAAPT | 0.560 [0.539, 0.580] | 0.842 | 1 | 13 | 3 |
| PENN | 0.560 [0.539, 0.579] | 0.720 | 1 | 13 | 3 |
| DIO | 0.560 [0.541, 0.577] | 0.710 | 1 | 13 | 3 |
| BasicPitch | 0.557 [0.539, 0.573] | 0.775 | 1 | 13 | 3 |
| pYIN | 0.506 [0.482, 0.525] | 0.656 | 0 | 17 | 0 |
| REAPER (crashed) | - | - | - | - | - |

Voicing F1 measures voiced/unvoiced detection, regardless of pitch accuracy.

Brackets show 95% confidence intervals. Beats and loses to count statistically significant wins and losses after adjusting for all pairwise comparisons. Undetermined means the data do not resolve the difference.

REAPER crashed in 56 of 90 runs and is unranked because its results are incomplete.

### Performance by dataset

| Tracker | APLAWD ↑ | AVID ↑ | Bach10Synth ↑ | FDA ↑ | KEELE ↑ | OSFGlottis ↑ | SVD ↑ | SpeechSynth ↑ | URMP ↑ | Vocadito ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SwiftF0 | **0.732** | 0.739 | **0.905** | 0.707 | 0.769 | 0.703 | **0.761** | 0.809 | **0.838** | 0.847 |
| RMVPE | 0.727 | **0.759** | 0.841 | **0.719** | **0.772** | **0.714** | 0.755 | **0.842** | 0.689 | **0.862** |
| FCPE | 0.687 | 0.696 | 0.859 | 0.654 | 0.720 | 0.661 | 0.688 | 0.767 | 0.763 | 0.786 |
| TorchCREPE | 0.597 | 0.636 | 0.870 | 0.603 | 0.660 | 0.636 | 0.660 | 0.725 | 0.737 | 0.789 |
| CREPE | 0.612 | 0.638 | 0.858 | 0.591 | 0.654 | 0.631 | 0.661 | 0.713 | 0.750 | 0.781 |
| PESTO | 0.612 | 0.649 | 0.832 | 0.586 | 0.640 | 0.619 | 0.664 | 0.694 | 0.747 | 0.759 |
| SHS | 0.594 | 0.615 | 0.815 | 0.574 | 0.627 | 0.608 | 0.600 | 0.688 | 0.745 | 0.706 |
| Praat | 0.598 | 0.628 | 0.753 | 0.608 | 0.664 | 0.599 | 0.620 | 0.688 | 0.652 | 0.696 |
| RAPT | 0.591 | 0.608 | 0.789 | 0.579 | 0.624 | 0.614 | 0.579 | 0.676 | 0.623 | 0.713 |
| HarmoF0 | 0.556 | 0.591 | 0.838 | 0.511 | 0.604 | 0.575 | 0.565 | 0.690 | 0.742 | 0.717 |
| SWIPE | 0.544 | 0.562 | 0.776 | 0.540 | 0.619 | 0.596 | 0.571 | 0.672 | 0.624 | 0.592 |
| SPICE | 0.482 | 0.528 | 0.756 | 0.497 | 0.549 | 0.579 | 0.565 | 0.651 | 0.681 | 0.734 |
| Harvest | 0.559 | 0.546 | 0.791 | 0.499 | 0.582 | 0.471 | 0.591 | 0.591 | 0.674 | 0.695 |
| YAAPT | 0.502 | 0.541 | 0.609 | 0.522 | 0.585 | 0.556 | 0.573 | 0.623 | 0.406 | 0.687 |
| PENN | 0.532 | 0.568 | 0.677 | 0.517 | 0.573 | 0.563 | 0.550 | 0.570 | 0.447 | 0.600 |
| DIO | 0.541 | 0.516 | 0.690 | 0.486 | 0.575 | 0.486 | 0.550 | 0.603 | 0.542 | 0.610 |
| BasicPitch | 0.367 | 0.451 | 0.875 | 0.422 | 0.483 | 0.499 | 0.473 | 0.566 | 0.798 | 0.638 |
| pYIN | 0.308 | 0.405 | 0.779 | 0.385 | 0.405 | 0.510 | 0.419 | 0.527 | 0.671 | 0.653 |
| REAPER (crashed) | - | - | - | - | - | - | - | - | - | - |

Pitch F1 averaged over the 8 scored conditions within each corpus, using the same threshold as the overall score. Rows follow the overall ranking.

### Performance by recording condition

| Tracker | Clean ↑ | level ↑ | scene ↑ | room ↑ | mic ↑ | scene+room ↑ | scene+mic ↑ | room+mic ↑ | scene+room+mic ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SwiftF0 | **0.949** | **0.956** | **0.754** | 0.784 | **0.956** | **0.636** | **0.751** | 0.783 | **0.628** |
| RMVPE | 0.943 | 0.945 | 0.749 | **0.786** | 0.944 | 0.604 | 0.740 | **0.785** | 0.591 |
| FCPE | 0.935 | 0.943 | 0.716 | 0.744 | 0.933 | 0.566 | 0.671 | 0.739 | 0.513 |
| TorchCREPE | 0.909 | 0.909 | 0.670 | 0.697 | 0.908 | 0.498 | 0.659 | 0.696 | 0.493 |
| CREPE | 0.914 | 0.914 | 0.657 | 0.696 | 0.914 | 0.497 | 0.651 | 0.693 | 0.489 |
| PESTO | 0.868 | 0.899 | 0.663 | 0.674 | 0.899 | 0.496 | 0.654 | 0.673 | 0.485 |
| SHS | 0.894 | 0.895 | 0.606 | 0.675 | 0.896 | 0.468 | 0.592 | 0.672 | 0.454 |
| Praat | 0.925 | 0.925 | 0.557 | 0.706 | 0.923 | 0.433 | 0.541 | 0.704 | 0.416 |
| RAPT | 0.891 | 0.902 | 0.555 | 0.687 | 0.900 | 0.428 | 0.545 | 0.686 | 0.413 |
| HarmoF0 | 0.843 | 0.843 | 0.580 | 0.680 | 0.843 | 0.466 | 0.566 | 0.680 | 0.452 |
| SWIPE | 0.855 | 0.861 | 0.563 | 0.623 | 0.862 | 0.403 | 0.553 | 0.621 | 0.391 |
| SPICE | 0.767 | 0.766 | 0.600 | 0.610 | 0.765 | 0.448 | 0.586 | 0.608 | 0.433 |
| Harvest | 0.826 | 0.827 | 0.558 | 0.611 | 0.824 | 0.418 | 0.544 | 0.612 | 0.404 |
| YAAPT | 0.781 | 0.782 | 0.494 | 0.593 | 0.786 | 0.382 | 0.480 | 0.602 | 0.364 |
| PENN | 0.854 | 0.857 | 0.427 | 0.645 | 0.865 | 0.321 | 0.420 | 0.641 | 0.303 |
| DIO | 0.880 | 0.880 | 0.436 | 0.637 | 0.880 | 0.303 | 0.421 | 0.634 | 0.288 |
| BasicPitch | 0.711 | 0.712 | 0.513 | 0.581 | 0.712 | 0.432 | 0.505 | 0.580 | 0.422 |
| pYIN | 0.678 | 0.678 | 0.407 | 0.605 | 0.679 | 0.347 | 0.394 | 0.604 | 0.336 |
| REAPER (crashed) | - | - | - | - | - | - | - | - | - |

Pitch F1 averaged equally over the corpora. Clean (`identity`) is the unmodified audio and is excluded from the overall score. `level` applies only level normalization. All scored conditions use the same normalization.

### Factor effects and interactions

| Tracker | scene | room | mic | scene x room | scene x mic | room x mic | scene x room x mic |
|---|---:|---:|---:|---:|---:|---:|---:|
| SwiftF0 | -0.178 | -0.147 | -0.003 | 0.052 | -0.005 | -0.003 | -0.006 |
| RMVPE | -0.194 | -0.153 | -0.006 | 0.012 | -0.010 | -0.003 | -0.003 |
| FCPE | -0.223 | -0.175 | -0.028 | 0.043 | -0.041 | -0.002 | -0.014 |
| TorchCREPE | -0.223 | -0.191 | -0.004 | 0.043 | -0.007 | 0.003 | 0.006 |
| CREPE | -0.231 | -0.190 | -0.004 | 0.058 | -0.006 | -0.003 | 0.001 |
| PESTO | -0.211 | -0.196 | -0.005 | 0.058 | -0.010 | -0.001 | -0.001 |
| SHS | -0.255 | -0.180 | -0.007 | 0.084 | -0.013 | -0.002 | 0.004 |
| Praat | -0.328 | -0.172 | -0.009 | 0.095 | -0.015 | -0.001 | -0.001 |
| RAPT | -0.308 | -0.173 | -0.007 | 0.085 | -0.011 | -0.002 | -0.006 |
| HarmoF0 | -0.245 | -0.139 | -0.007 | 0.049 | -0.015 | -0.000 | -0.001 |
| SWIPE | -0.264 | -0.200 | -0.006 | 0.078 | -0.010 | -0.003 | 0.002 |
| SPICE | -0.171 | -0.155 | -0.008 | 0.004 | -0.014 | -0.000 | -0.001 |
| Harvest | -0.238 | -0.177 | -0.008 | 0.074 | -0.014 | 0.002 | -0.004 |
| YAAPT | -0.261 | -0.150 | -0.004 | 0.072 | -0.022 | 0.000 | -0.009 |
| PENN | -0.384 | -0.165 | -0.005 | 0.107 | -0.015 | -0.010 | 0.001 |
| DIO | -0.395 | -0.189 | -0.008 | 0.112 | -0.013 | -0.001 | 0.002 |
| BasicPitch | -0.178 | -0.107 | -0.004 | 0.050 | -0.008 | -0.002 | -0.002 |
| pYIN | -0.270 | -0.066 | -0.006 | 0.015 | -0.012 | 0.000 | 0.002 |
| REAPER (crashed) | - | - | - | - | - | - | - |

The scene, room and mic columns show the change in pitch F1 when that factor is enabled, averaged over all settings of the other factors. Negative values mean a loss. For pair interactions, a negative value means the combined loss exceeds the sum of the separate losses. A positive value means the combined loss is smaller. Pair interactions are averaged over both settings of the third factor. The three-factor interaction shows how the scene × room interaction changes when mic is enabled.

### Pitch accuracy

| Tracker | <10c ↑ | <25c ↑ | <50c ↑ | <200c ↑ | Octave up ↓ | Octave down ↓ |
|---|---:|---:|---:|---:|---:|---:|
| SwiftF0 | 67.7 | **84.8** | **91.4** | 95.6 | 0.34 | 0.42 |
| RMVPE | **68.5** | 83.3 | 89.8 | 94.6 | 0.76 | 0.52 |
| FCPE | 66.0 | 82.1 | 89.3 | 94.8 | 0.73 | 1.02 |
| TorchCREPE | 46.4 | 80.8 | **91.4** | **96.6** | 0.97 | **0.18** |
| CREPE | 63.9 | 81.4 | 88.6 | 93.6 | 1.53 | 0.37 |
| PESTO | 38.3 | 71.3 | 81.5 | 87.7 | 3.39 | 0.80 |
| SHS | 58.7 | 79.0 | 85.9 | 91.5 | 1.43 | 1.65 |
| Praat | 57.1 | 70.7 | 78.0 | 85.1 | 0.40 | 4.47 |
| RAPT | 56.4 | 71.2 | 79.3 | 87.6 | 0.37 | 3.95 |
| HarmoF0 | 55.7 | 80.0 | 87.6 | 93.9 | **0.15** | 1.27 |
| SWIPE | 49.6 | 70.2 | 80.9 | 90.6 | 1.19 | 1.62 |
| SPICE | 51.4 | 72.9 | 84.5 | 95.1 | 0.88 | 0.23 |
| Harvest | 56.2 | 71.8 | 81.0 | 90.6 | 0.64 | 0.68 |
| YAAPT | 10.0 | 34.6 | 64.4 | 79.7 | 1.05 | 6.93 |
| PENN | 47.4 | 68.7 | 79.6 | 89.0 | 1.22 | 1.55 |
| DIO | 53.5 | 70.2 | 79.5 | 88.6 | 1.67 | 1.07 |
| BasicPitch | 24.8 | 56.3 | 78.4 | 90.2 | 0.85 | 0.40 |
| pYIN | 52.2 | 71.5 | 82.4 | 92.8 | 0.54 | 1.79 |
| REAPER (crashed) | - | - | - | - | - | - |

Percentages among frames where both the tracker and reference are voiced and the reference pitch is verified and within the search range. Counts are pooled across all scored conditions and corpora at each tracker's selected threshold. Octave up and down count errors 1100–1300 cents above and below the reference pitch, respectively.

BasicPitch uses its note output, which has semitone resolution.

## Properties

Frame alignment and runtime do not contribute to the overall score.

### Frame alignment (chirp probe, ms)

| Tracker | Worst measured alignment error (ms) ↓ |
|---|---:|
| Praat | **0.01** |
| DIO | 0.05 |
| RAPT | 0.18 |
| Harvest | 0.21 |
| RMVPE | 0.30 |
| SHS | 0.41 |
| CREPE | 0.79 |
| pYIN | 0.80 |
| SwiftF0 | 0.92 |
| FCPE | 1.34 |
| TorchCREPE | 1.49 |
| SWIPE | 1.70 |
| SPICE | 2.23 |
| PESTO | 2.34 |
| YAAPT | 3.22 |
| HarmoF0 | 4.11 |
| BasicPitch | 9.85 |
| PENN | 11.46 |
| REAPER (crashed) | - |

Largest absolute time offset measured across chirp bands. A larger offset (for example, above 2 ms) can indicate an algorithm error or a model that learned to place pitch estimates too early or too late from misaligned training labels.

### Speed

| Tracker | Speed (× real time, one core) ↑ |
|---|---:|
| RAPT | **1276.6** ± 14.8 |
| SHS | 1080.3 ± 9.1 |
| Praat | 525.3 ± 17.9 |
| DIO | 211.9 ± 11.5 |
| SwiftF0 | 179.6 ± 6.7 |
| SPICE | 86.2 ± 2.7 |
| REAPER (crashed) | 68.7 ± 0.7 |
| BasicPitch | 59.7 ± 0.7 |
| YAAPT | 50.5 ± 0.4 |
| SWIPE | 50.3 ± 0.2 |
| FCPE | 27.8 ± 3.2 |
| PESTO | 20.0 ± 0.1 |
| Harvest | 18.4 ± 0.4 |
| RMVPE | 13.6 ± 0.8 |
| pYIN | 9.6 ± 0.6 |
| PENN | 4.7 ± 0.1 |
| HarmoF0 | 4.1 ± 0.1 |
| CREPE | 0.4 ± 0.0 |
| TorchCREPE | 0.4 ± 0.0 |

Audio duration divided by the median CPU time over five rounds: 20× means 20 seconds of audio processed per second of CPU time. Every tracker runs pinned to one CPU core, with the thread limits set to 1, after a warm-up, and the order of the trackers is shuffled in each round. The value after ± is the standard deviation over the rounds. A tracker without five successful rounds shows -. CREPE also waited for its own threads, so its wall-clock time on one core is longer. Speed is measured separately by [speed.py](speed.py), including for trackers with incomplete accuracy results. CPU: AMD Ryzen 9 8945HS w/ Radeon 780M Graphics.

<!-- /report -->
