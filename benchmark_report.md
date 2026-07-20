# Pitch Benchmark Report

## Leaderboard

Overall = harmonic mean of the seven track scores, equal weights. A missing track makes the Overall n/a (never a silent partial mean); a zero track sinks it to 0, annotated with the failing track; zero-Overall trackers are ordered among themselves by the harmonic mean of their non-zero tracks. `†` = fixed operating point (no confidence curve to tune).

| **Algorithm** | **Overall** | **Accuracy** | **Noise** | **Signals** | **Stability** | **Dynamics** | **Notes** | **Speed** |
|---|---|---|---|---|---|---|---|---|
| Praat | **0.871** | 0.827 | 0.666 | **0.953** | **0.995** | **0.987** | **0.793** | **0.997** |
| RMVPE | 0.784 | **0.845** | **0.895** | 0.471 | 0.993 | 0.949 | 0.785 | 0.853 |

### Operating points (theta\*)

| **Algorithm** | **theta\*** | **selected on** | **#datasets** |
|---|---|---|---|
| Praat | 0.6 | clean-full | 17 |
| RMVPE | 0.5 | clean-full | 17 |

## Methodology

### The contract
An algorithm is `(f0(t), q(t))`: a pitch estimate and a voicing confidence per frame.
`f0 <= 0` or `NaN` is a **voicing claim, never a pitch estimate**: at threshold theta, a
frame counts as voiced-with-pitch iff `q >= theta` and f0 is finite-positive. Abstaining on a truly
voiced frame costs recall; it never fabricates a cents error.

### The core metric: the pitch F-score
With counts at threshold theta and cents tolerance T -- `n_ok(T)` = frames truly voiced,
voiced by the tracker, and within T cents; `tp/fp/fn` = the voicing confusion --

    pitch recall    R(T) = n_ok / (tp + fn)
    pitch precision P(T) = n_ok / (tp + fp)
    pitch F-score   F(T) = 2 n_ok / ((tp+fp) + (tp+fn))

Pure counts, no weights. The identity `R = voicing_recall x accuracy_on_voiced`
displays both classic RPA definitions as factors of one product. `T -> inf` recovers
voicing F1; the headline tolerance is 50 cents; the **tolerance AUC** integrates F over
T in [0, 100] and equals `1 - truncated-MAE/100` on the scored frames.

### The operating point
One global **theta\*** per algorithm: argmax over the threshold grid of the
equal-per-dataset mean F@50 on clean cells (full cells preferred, probe fallback
stamped; ties -> lowest theta). Frozen; every track reads it. There is no per-cell
threshold selection anywhere.

### The tracks (questions, not decompositions)
Each column is a complete situation of use, so nothing is scored twice. Diagnostics
(voicing P/R/F1, accuracy-on-voiced, RPA@50, cents, RCA, octave/gross rates,
per-band, smoothness, latency, coverage) explain the columns and are never scored.

1. **Accuracy** -- how correct is the output curve on clean real recordings? (F@50)
2. **Noise** -- how much survives corruption? (F@50 degraded / clean, paired probe clips)
3. **Signals** -- how much of the signal-class space is handled? (mean over stationary families; worst family named)
4. **Stability** -- does one threshold work everywhere? (F(theta*)/F(oracle))
5. **Dynamics** -- is a moving pitch followed faithfully? (steady jitter + vibrato retention)
6. **Notes** -- is musical structure recoverable? (COnP)
7. **Speed** -- is it deployable? (1/(1+RTF))

### Weight audit
The only named conventions: tolerances (10/25/50 cents), T_MAX=100,
steady-jitter normalizer sigma0=10 cents, speed mapping 1/(1+RTF), the HM's
equal track weights, and equal-per-dataset pooling for track scores (the frame-pooled
variant is shown as a diagnostic). Everything else is counts.

### Statistics
All CIs are 95% paired cluster bootstraps over per-clip sufficient statistics (clusters
= speaker/singer/piece), and every scored value is exactly recomputable from summed
per-clip stats -- the CI machinery resamples and recomputes the same formula it reports.

## Datasets

| **Dataset** | **Domain** | **Clips** | **Hours** | **Avg len (s)** | **Voiced %** | **f0 p5-p50-p95 (Hz)** | **Band coverage** |
|---|---|---|---|---|---|---|---|
| NSynth | Music | 3319 | 3.7 | 4.0 | 53 | 73-277-1480 | bass 8%, low 39%, mid 29%, high 12%, vhigh 12% |
| PTDB | Speech | 4718 | 9.6 | 7.3 | 26 | 84-156-236 | bass 3%, low 95%, mid 2% |
| MIR1K | Music | 1000 | 2.2 | 8.0 | 70 | 121-231-390 | low 63%, mid 37% |
| MDBStemSynth | Music | 230 | 15.6 | 243.6 | 40 | 73-209-704 | bass 11%, low 50%, mid 32%, high 6% |
| Vocadito | Music | 40 | 0.2 | 20.4 | 66 | 113-219-364 | low 69%, mid 31% |
| Bach10Synth | Music | 40 | 0.4 | 33.4 | 92 | 111-296-579 | low 38%, mid 59%, high 2% |
| SpeechSynth | Speech | 219 | 0.1 | 2.0 | 47 | 126-213-273 | low 90%, mid 10% |
| KEELE | Speech | 10 | 0.1 | 33.7 | 54 | 81-173-295 | bass 5%, low 85%, mid 10% |
| FDA | Speech | 100 | 0.1 | 3.3 | 42 | 91-205-296 | low 80%, mid 20% |
| MOCHA | Speech | 3690 | 4.3 | 4.2 | 46 | 98-188-355 | low 82%, mid 17% |
| CMUArctic | Speech | 3377 | 2.8 | 3.0 | 60 | 93-130-195 | bass 1%, low 99% |
| SVD | Speech | 634 | 0.3 | 1.9 | 73 | 87-181-287 | bass 3%, low 86%, mid 12% |
| APLAWD | Speech | 10979 | 2.7 | 0.9 | 58 | 81-134-251 | bass 4%, low 91%, mid 4% |
| OSFGlottis (sparse-voiced) | Speech | 14 | 6.9 | 1764.6 | 20 | 113-188-257 | low 96%, mid 4% |
| AVID (sparse-voiced) | Speech | 51 | 18.3 | 1290.7 | 21 | 114-208-346 | low 76%, mid 24% |
| M4Singer (voicing-only GT) | Music | 20896 | 29.7 | 5.1 | 86 | 92-233-523 | bass 1%, low 55%, mid 43% |
| URMP | Music | 149 | 4.6 | 111.0 | 76 | 109-348-856 | bass 1%, low 31%, mid 54%, high 12%, vhigh 2% |

Bands: bass (<80 Hz), low (80-260 Hz), mid (260-650 Hz), high (650-1050 Hz), vhigh (>=1050 Hz). f0 statistics are over in-window voiced frames.

## Track 1: Accuracy

Clean real recordings at each algorithm's theta\*. Datasets flagged voicing-only are excluded from the score; sparse-voiced corpora are flagged (precision denominators dominated by silence).

### Praat

| **Dataset** | **F@50 [95% CI]** | **R@50** | **P@50** | **AUC** | **RPA@50 (voiced)** | **Coverage** |
|---|---|---|---|---|---|---|
| APLAWD | 0.802 [0.780, 0.821] | 0.756 | 0.854 | 0.720 | 0.957 | 0.790 |
| AVID (sparse) | 0.646 [0.619, 0.672] | 0.878 | 0.511 | 0.590 | 0.948 | 0.926 |
| Bach10Synth | 0.989 [0.988, 0.990] | 0.980 | 0.998 | 0.979 | 0.998 | 0.983 |
| CMUArctic | 0.827 [0.800, 0.858] | 0.790 | 0.867 | 0.748 | 0.939 | 0.841 |
| FDA | 0.823 [0.805, 0.839] | 0.751 | 0.911 | 0.746 | 0.951 | 0.790 |
| KEELE | 0.776 [0.731, 0.807] | 0.710 | 0.855 | 0.705 | 0.950 | 0.747 |
| M4Singer (voicing-only, unscored) | 0.636 [0.614, 0.662] | 0.580 | 0.703 | 0.546 | 0.703 | 0.825 |
| MDBStemSynth | 0.959 [0.949, 0.966] | 0.949 | 0.970 | 0.935 | 0.988 | 0.960 |
| MIR1K | 0.952 [0.943, 0.960] | 0.941 | 0.963 | 0.885 | 0.981 | 0.960 |
| MOCHA | 0.698 [0.645, 0.749] | 0.639 | 0.768 | 0.629 | 0.922 | 0.693 |
| NSynth | 0.833 [0.759, 0.896] | 0.749 | 0.940 | 0.796 | 0.940 | 0.797 |
| OSFGlottis (sparse) | 0.751 [0.608, 0.847] | 0.898 | 0.646 | 0.715 | 0.962 | 0.933 |
| PTDB | 0.765 [0.728, 0.793] | 0.703 | 0.837 | 0.689 | 0.941 | 0.748 |
| SVD | 0.792 [0.787, 0.798] | 0.775 | 0.811 | 0.726 | 0.920 | 0.842 |
| SpeechSynth | 0.953 [0.944, 0.960] | 0.933 | 0.973 | 0.897 | 0.991 | 0.941 |
| URMP | 0.930 [0.913, 0.944] | 0.948 | 0.912 | 0.911 | 0.962 | 0.985 |
| Vocadito | 0.924 [0.871, 0.954] | 0.932 | 0.916 | 0.899 | 0.970 | 0.961 |

Factorization (frame-pooled): pitch recall 0.721 = voicing recall 0.870 x accuracy-on-voiced 0.828; pitch F 0.739.

### RMVPE

| **Dataset** | **F@50 [95% CI]** | **R@50** | **P@50** | **AUC** | **RPA@50 (voiced)** | **Coverage** |
|---|---|---|---|---|---|---|
| APLAWD | 0.855 [0.833, 0.875] | 0.778 | 0.950 | 0.795 | 0.986 | 0.789 |
| AVID (sparse) | 0.677 [0.647, 0.706] | 0.920 | 0.535 | 0.635 | 0.981 | 0.938 |
| Bach10Synth | 0.988 [0.984, 0.991] | 0.980 | 0.996 | 0.974 | 0.999 | 0.981 |
| CMUArctic | 0.876 [0.852, 0.894] | 0.819 | 0.941 | 0.817 | 0.976 | 0.840 |
| FDA | 0.870 [0.867, 0.874] | 0.807 | 0.944 | 0.810 | 0.981 | 0.823 |
| KEELE | 0.808 [0.764, 0.841] | 0.741 | 0.889 | 0.748 | 0.971 | 0.763 |
| M4Singer (voicing-only, unscored) | 0.632 [0.612, 0.656] | 0.575 | 0.702 | 0.544 | 0.702 | 0.818 |
| MDBStemSynth | 0.925 [0.907, 0.937] | 0.920 | 0.930 | 0.905 | 0.993 | 0.926 |
| MIR1K | 0.960 [0.955, 0.965] | 0.955 | 0.966 | 0.884 | 0.984 | 0.970 |
| MOCHA | 0.753 [0.704, 0.806] | 0.651 | 0.894 | 0.695 | 0.969 | 0.672 |
| NSynth | 0.769 [0.709, 0.817] | 0.666 | 0.911 | 0.729 | 0.940 | 0.709 |
| OSFGlottis (sparse) | 0.791 [0.648, 0.882] | 0.925 | 0.691 | 0.758 | 0.986 | 0.937 |
| PTDB | 0.802 [0.769, 0.829] | 0.742 | 0.872 | 0.740 | 0.969 | 0.766 |
| SVD | 0.843 [0.838, 0.848] | 0.783 | 0.912 | 0.789 | 0.965 | 0.811 |
| SpeechSynth | 0.966 [0.960, 0.970] | 0.973 | 0.959 | 0.924 | 0.997 | 0.976 |
| URMP | 0.896 [0.875, 0.914] | 0.848 | 0.949 | 0.868 | 0.969 | 0.875 |
| Vocadito | 0.956 [0.949, 0.962] | 0.975 | 0.938 | 0.910 | 0.990 | 0.985 |

Factorization (frame-pooled): pitch recall 0.713 = voicing recall 0.849 x accuracy-on-voiced 0.840; pitch F 0.740.


## Track 2: Noise robustness

Retention ratio F@50(degraded)/F@50(clean) on the SAME probe clips, at theta\*, equal-per-dataset. **Floor-effect caveat**: a tracker that is already poor on clean has little to lose; read this column next to Accuracy, never alone.

| **Algorithm** | **Mean** | **chime** | **codec** | **demand** | **fade** | **gain-40** | **pink** | **pink_snr+10** | **pink_snr-5** | **reverb** | **room** | **telephone** | **white** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Praat | 0.666 | 0.438 | 0.999 | 0.494 | 0.820 | 1.000 | 0.439 | 0.977 | 0.055 | 0.737 | 0.722 | 0.954 | 0.356 |
| RMVPE | 0.895 | 0.838 | 1.002 | 0.896 | 0.966 | 0.922 | 0.942 | 0.993 | 0.796 | 0.766 | 0.728 | 0.901 | 0.988 |

By condition family:

| **Algorithm** | **additive** | **convolutional** | **filtering** | **dynamics** | **snr** |
|---|---|---|---|---|---|
| Praat | 0.432 | 0.729 | 0.976 | 0.910 | 0.516 |
| RMVPE | 0.916 | 0.747 | 0.951 | 0.944 | 0.895 |

## Track 3: Signal robustness

Stationary synthetic families with exact labels; per-family accuracy = coverage-aware pitch recall@50 at theta\*. Score = the MEAN over families (each family is one probe question, equally weighted); the worst family is named beside it as the diagnostic. A worst-family SCORE was measured and rejected: 6 of 7 surveyed trackers have at least one exactly-dead family, so a min would zero almost the whole field. Controls (no pitch present) report false-positive rate as a diagnostic.

| **Algorithm** | **Score (worst)** | **glide** | **harm_bass** | **harm_high** | **harm_level** | **harm_low** | **harm_mid** | **harm_vhigh** | **interference** | **irn** | **missing_f0** | **sine_bass** | **sine_high** | **sine_level** | **sine_low** | **sine_mid** | **sine_vhigh** | **tilt_high** | **tilt_low** | **tilt_mid** | **tilt_vhigh** | **unresolved** | **vibrato_fast** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Praat | **0.95** (worst interference 0.65) | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.65 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 |
| RMVPE | **0.47** (worst harm_vhigh 0.00) | 1.00 | 0.99 | 0.58 | 0.25 | 0.94 | 0.85 | 0.00 | 1.00 | 0.57 | 0.61 | 0.00 | 0.24 | 0.29 | 0.35 | 0.57 | 0.00 | 0.37 | 0.31 | 0.43 | 0.00 | 0.00 | 1.00 |

Controls (false-positive rate, lower is better; diagnostic only):

| **Algorithm** | **noise** | **silence** | **whisper** |
|---|---|---|---|
| Praat | 0.000 | 0.000 | 0.000 |
| RMVPE | 0.000 | 0.000 | 0.044 |

## Track 4: Operating stability

F@50 at the frozen theta\* divided by F@50 at each cell's oracle threshold (1.0 = the one global threshold loses nothing anywhere). Fixed-operating-point trackers are trivially 1.0 (nothing to mistune) and flagged `†` on the leaderboard.

| **Algorithm** | **Efficiency** | **Worst peak-vs-theta\* gap** |
|---|---|---|
| Praat | 0.995 | 0.016 |
| RMVPE | 0.993 | 0.033 |

## Track 5: Tracking dynamics

Trajectory families with exact labels, read at theta\*. Steady tones: jitter (cents std) and bias, scored sigma0/(sigma0+jitter) with sigma0=10c. Vibrato: modulation-depth retention x voiced coverage. Track score = mean of the two family groups.

| **Algorithm** | **Score** | **steady_harm** | **steady_sine** | **vib_half** | **vib_one** | **vib_two** |
|---|---|---|---|---|---|---|
| Praat | 0.987 | jit 0.0c, bias +0.0c | jit 0.0c, bias -0.0c | ret 0.96 x cov 1.00 | ret 0.97 x cov 1.00 | ret 0.99 x cov 1.00 |
| RMVPE | 0.949 | jit 0.7c, bias -0.8c | jit 0.8c, bias -0.3c | ret 0.98 x cov 1.00 | ret 0.98 x cov 1.00 | ret 0.94 x cov 1.00 |

## Track 6: Notes

Note transcription (COnP / COnPOff). This track selects its own threshold and segmentation cost internally -- the one deliberate exception to the global-theta rule, documented here.

| **Algorithm** | **M4Singer** (COnP/COnPOff) | **URMP** (COnP/COnPOff) | **Vocadito** (COnP/COnPOff) |
|---|---|---|---|
| Praat | n/a | 0.829 / 0.756 | 0.758 / 0.629 |
| RMVPE | n/a | 0.789 / 0.689 | 0.780 / 0.646 |

## Track 7: Speed

| **Algorithm** | **RTF (cpu)** | **Score 1/(1+RTF)** |
|---|---|---|
| Praat | 0.003 | 0.997 |
| RMVPE | 0.172 | 0.853 |

## Caveats

- **Floor effect** (Noise track): retention is only meaningful next to absolute
  Accuracy; a tracker that is poor on clean audio has little room to drop.
- **Sparse-voiced corpora** (OSFGlottis, AVID): mostly-unvoiced sessions; precision
  denominators are dominated by silence false positives. Flagged in the tables.
- **Score-grade ground truth** (M4Singer): notated pitch, not performed f0; voicing GT
  is reliable, pitch scores are not; excluded from Accuracy.
- **Training-data leakage**: learned trackers may have trained on these public corpora.
  A clean-only advantage that collapses under degradation is a leakage signature;
  degraded and synthetic tracks move inputs away from anything seen verbatim.
- **Why v1 scored pitch conditionally, and v2 does not**: v1 computed RPA only where
  both sides agreed on voicing, to dodge GT voicing-label errors and to accommodate
  trackers that output no pitch on unvoiced frames. v2 scores the joint event instead:
  labels are now exact-by-construction or consensus-derived, paired comparisons cancel
  the shared residue, and the (f0, q) contract makes abstention a voicing claim (a
  recall cost) rather than a fabricated pitch error. The conditional quantity survives
  as the accuracy-on-voiced diagnostic in the factorization.

## Appendix

### Accuracy by dataset group

**By Origin**

| **Algorithm** | **Synthetic** | **Real** |
|---|---|---|
| Praat | 0.934 | 0.807 |
| RMVPE | 0.912 | 0.841 |

**By Domain**

| **Algorithm** | **Speech** | **Music** |
|---|---|---|
| Praat | 0.783 | 0.931 |
| RMVPE | 0.824 | 0.916 |

**By Cross-Dimension**

| **Algorithm** | **Synthetic + Speech** | **Synthetic + Music** | **Real + Speech** | **Real + Music** |
|---|---|---|---|---|
| Praat | 0.953 | 0.927 | 0.764 | 0.935 |
| RMVPE | 0.966 | 0.894 | 0.808 | 0.937 |

### Aggregation sensitivity

The Overall uses the harmonic mean (one mean family everywhere; dominated by the weakest track). The alternatives on the same track scores:

| **Algorithm** | **AM** | **GM** | **HM (used)** |
|---|---|---|---|
| Praat | 0.888 | 0.880 | 0.871 |
| RMVPE | 0.827 | 0.808 | 0.784 |
