# Pitch Benchmark Report

## Leaderboard

**Notes is not measured in this run**, so Overall is not computed; rank by Correctness. Definitions for every column are in [METRICS.md](METRICS.md).

| **Algorithm** | **Overall** | **theta\*** | **Correctness** | **Noise** | **Signal types** | **Tracking** | **Speed** |
|---|---|---|---|---|---|---|---|
| Praat | n/a | 0.7 | 0.824 | 0.492 | **0.953** | **0.987** | **0.998** |
| CREPE | n/a | 0.6 | 0.825 | 0.637 | 0.893 | 0.914 | 0.718 |
| SwiftF0 | n/a | 0.9 | 0.774 | 0.525 | 0.817 | 0.881 | 0.955 |
| RMVPE | n/a | 0.5 | **0.853** | **0.756** | 0.471 | 0.940 | 0.900 |
| BasicPitch | n/a | 0.3 | 0.666 | 0.531 | 0.809 | 0.833 | 0.970 |
| DIO | n/a | 0.3 | 0.804 | 0.443 | 0.685 | 0.975 | 0.996 |
| RAPT | n/a | 0.2 | 0.816 | 0.400 | 0.664 | 0.953 | 0.998 |
| Harvest | n/a | 1.0 | 0.747 | 0.542 | 0.510 | 0.747 | 0.966 |
| SWIPE | n/a | 0.7 | 0.686 | 0.363 | 0.809 | 0.785 | 0.982 |
| SPICE | n/a | 0.8 | 0.706 | 0.504 | 0.507 | 0.729 | 0.955 |
| PENN | n/a | 0.3 | 0.791 | 0.426 | 0.675 | 0.870 | 0.630 |
| pYIN | n/a | 0.1 | 0.635 | 0.347 | 0.776 | 0.506 | 0.860 |
| TorchCREPE | n/a | 0.6 | 0.801 | 0.583 | 0.892 | 0.544 | 0.295 |
| YAAPT | n/a | 0.9 | 0.705 | 0.496 | 0.399 | 0.192 | 0.980 |
| REAPER | n/a | 0.1 | 0.504 | 0.295 | 0.222 | 0.317 | 0.000 |

**Praat** leads overall; RMVPE leads Correctness; RMVPE leads Noise.

## Track 1: Correctness

Pitch F on clean real recordings at each algorithm's theta\*, averaged with equal weight per dataset. Datasets whose pitch ground truth is score-grade are excluded by the scorer; sparse-voiced corpora are flagged in place.

### BasicPitch

| **Dataset** | **pitch F1 [95% CI]** | **pitch recall** | **pitch precision** |
|---|---|---|---|
| APLAWD [capped n30/t10.0] | 0.468 [0.387, 0.505] | 0.395 | 0.575 |
| AVID (sparse) [capped n30/t10.0] | 0.379 [0.320, 0.434] | 0.527 | 0.296 |
| Bach10Synth [capped n30/t10.0] | 0.953 [0.948, 0.959] | 0.947 | 0.960 |
| CMUArctic [capped n30/t10.0] | 0.559 [0.489, 0.655] | 0.516 | 0.611 |
| FDA [capped n30/t10.0] | 0.556 [0.470, 0.623] | 0.496 | 0.632 |
| KEELE [capped n30/t10.0] | 0.533 [0.475, 0.585] | 0.463 | 0.628 |
| MDBStemSynth [capped n30/t10.0] | 0.836 [0.783, 0.882] | 0.828 | 0.845 |
| MIR1K [capped n30/t10.0] | 0.803 [0.781, 0.820] | 0.797 | 0.809 |
| MOCHA [capped n30/t10.0] | 0.510 [0.458, 0.551] | 0.446 | 0.594 |
| NSynth [capped n30/t10.0] | 0.942 [0.871, 0.982] | 0.952 | 0.932 |
| OSFGlottis (sparse) [capped n30/t10.0] | 0.598 [0.456, 0.692] | 0.784 | 0.484 |
| PTDB [capped n30/t10.0] | 0.485 [0.445, 0.523] | 0.421 | 0.572 |
| SVD [capped n30/t10.0] | 0.560 [0.519, 0.597] | 0.491 | 0.650 |
| SpeechSynth [capped n30/t10.0] | 0.756 [0.718, 0.791] | 0.735 | 0.779 |
| URMP [capped n30/t10.0] | 0.925 [0.904, 0.943] | 0.943 | 0.908 |
| Vocadito [capped n30/t10.0] | 0.793 [0.765, 0.817] | 0.802 | 0.785 |

Track score (equal per dataset): **0.666** [0.648, 0.679].

Where the recall comes from: pitch recall 0.779 = voicing recall 0.891 x the correct rate (exact+close, below).

### CREPE

| **Dataset** | **pitch F1 [95% CI]** | **pitch recall** | **pitch precision** |
|---|---|---|---|
| APLAWD [capped n30/t10.0] | 0.730 [0.680, 0.793] | 0.620 | 0.889 |
| AVID (sparse) [capped n30/t10.0] | 0.492 [0.420, 0.568] | 0.796 | 0.356 |
| Bach10Synth [capped n30/t10.0] | 0.982 [0.977, 0.986] | 0.975 | 0.988 |
| CMUArctic [capped n30/t10.0] | 0.820 [0.774, 0.866] | 0.748 | 0.907 |
| FDA [capped n30/t10.0] | 0.815 [0.790, 0.836] | 0.759 | 0.879 |
| KEELE [capped n30/t10.0] | 0.784 [0.741, 0.818] | 0.712 | 0.871 |
| MDBStemSynth [capped n30/t10.0] | 0.908 [0.854, 0.950] | 0.939 | 0.878 |
| MIR1K [capped n30/t10.0] | 0.946 [0.940, 0.956] | 0.939 | 0.954 |
| MOCHA [capped n30/t10.0] | 0.721 [0.653, 0.776] | 0.636 | 0.831 |
| NSynth [capped n30/t10.0] | 0.940 [0.892, 0.970] | 0.949 | 0.931 |
| OSFGlottis (sparse) [capped n30/t10.0] | 0.717 [0.536, 0.826] | 0.889 | 0.601 |
| PTDB [capped n30/t10.0] | 0.721 [0.683, 0.754] | 0.665 | 0.788 |
| SVD [capped n30/t10.0] | 0.808 [0.787, 0.828] | 0.729 | 0.906 |
| SpeechSynth [capped n30/t10.0] | 0.965 [0.951, 0.975] | 0.961 | 0.968 |
| URMP [capped n30/t10.0] | 0.923 [0.897, 0.946] | 0.906 | 0.942 |
| Vocadito [capped n30/t10.0] | 0.933 [0.917, 0.945] | 0.966 | 0.901 |

Track score (equal per dataset): **0.825** [0.810, 0.837].

Where the recall comes from: pitch recall 0.890 = voicing recall 0.925 x the correct rate (exact+close, below).

### DIO

| **Dataset** | **pitch F1 [95% CI]** | **pitch recall** | **pitch precision** |
|---|---|---|---|
| APLAWD [capped n30/t10.0] | 0.816 [0.747, 0.843] | 0.757 | 0.885 |
| AVID (sparse) [capped n30/t10.0] | 0.479 [0.405, 0.546] | 0.743 | 0.353 |
| Bach10Synth [capped n30/t10.0] | 0.987 [0.977, 0.993] | 0.980 | 0.994 |
| CMUArctic [capped n30/t10.0] | 0.793 [0.739, 0.831] | 0.747 | 0.844 |
| FDA [capped n30/t10.0] | 0.772 [0.758, 0.785] | 0.722 | 0.830 |
| KEELE [capped n30/t10.0] | 0.758 [0.719, 0.792] | 0.703 | 0.822 |
| MDBStemSynth [capped n30/t10.0] | 0.938 [0.896, 0.969] | 0.946 | 0.931 |
| MIR1K [capped n30/t10.0] | 0.922 [0.910, 0.936] | 0.911 | 0.934 |
| MOCHA [capped n30/t10.0] | 0.694 [0.639, 0.742] | 0.629 | 0.776 |
| NSynth [capped n30/t10.0] | 0.915 [0.851, 0.956] | 0.909 | 0.920 |
| OSFGlottis (sparse) [capped n30/t10.0] | 0.705 [0.543, 0.806] | 0.904 | 0.579 |
| PTDB [capped n30/t10.0] | 0.712 [0.677, 0.745] | 0.666 | 0.765 |
| SVD [capped n30/t10.0] | 0.775 [0.752, 0.796] | 0.699 | 0.871 |
| SpeechSynth [capped n30/t10.0] | 0.938 [0.925, 0.950] | 0.920 | 0.957 |
| URMP [capped n30/t10.0] | 0.764 [0.661, 0.851] | 0.693 | 0.851 |
| Vocadito [capped n30/t10.0] | 0.898 [0.845, 0.929] | 0.906 | 0.891 |

Track score (equal per dataset): **0.804** [0.789, 0.816].

Where the recall comes from: pitch recall 0.844 = voicing recall 0.892 x the correct rate (exact+close, below).

### Harvest

| **Dataset** | **pitch F1 [95% CI]** | **pitch recall** | **pitch precision** |
|---|---|---|---|
| APLAWD [capped n30/t10.0] | 0.751 [0.706, 0.771] | 0.805 | 0.704 |
| AVID (sparse) [capped n30/t10.0] | 0.338 [0.284, 0.393] | 0.824 | 0.213 |
| Bach10Synth [capped n30/t10.0] | 0.998 [0.997, 0.998] | 0.998 | 0.997 |
| CMUArctic [capped n30/t10.0] | 0.731 [0.696, 0.746] | 0.840 | 0.647 |
| FDA [capped n30/t10.0] | 0.673 [0.658, 0.690] | 0.828 | 0.566 |
| KEELE [capped n30/t10.0] | 0.664 [0.627, 0.696] | 0.772 | 0.583 |
| MDBStemSynth [capped n30/t10.0] | 0.956 [0.923, 0.982] | 0.987 | 0.928 |
| MIR1K [capped n30/t10.0] | 0.876 [0.863, 0.892] | 0.940 | 0.819 |
| MOCHA [capped n30/t10.0] | 0.576 [0.513, 0.635] | 0.679 | 0.499 |
| NSynth [capped n30/t10.0] | 0.891 [0.803, 0.947] | 0.914 | 0.869 |
| OSFGlottis (sparse) [capped n30/t10.0] | 0.522 [0.367, 0.636] | 0.933 | 0.362 |
| PTDB [capped n30/t10.0] | 0.581 [0.533, 0.625] | 0.762 | 0.469 |
| SVD [capped n30/t10.0] | 0.753 [0.723, 0.779] | 0.820 | 0.695 |
| SpeechSynth [capped n30/t10.0] | 0.881 [0.858, 0.903] | 0.957 | 0.816 |
| URMP [capped n30/t10.0] | 0.904 [0.873, 0.931] | 0.950 | 0.862 |
| Vocadito [capped n30/t10.0] | 0.863 [0.843, 0.881] | 0.974 | 0.775 |

Track score (equal per dataset): **0.747** [0.733, 0.758].

Where the recall comes from: pitch recall 0.925 = voicing recall 0.990 x the correct rate (exact+close, below).

### PENN

| **Dataset** | **pitch F1 [95% CI]** | **pitch recall** | **pitch precision** |
|---|---|---|---|
| APLAWD [capped n30/t10.0] | 0.651 [0.604, 0.696] | 0.516 | 0.882 |
| AVID (sparse) [capped n30/t10.0] | 0.562 [0.490, 0.627] | 0.793 | 0.435 |
| Bach10Synth [capped n30/t10.0] | 0.968 [0.964, 0.972] | 0.967 | 0.969 |
| CMUArctic [capped n30/t10.0] | 0.861 [0.840, 0.877] | 0.839 | 0.884 |
| FDA [capped n30/t10.0] | 0.777 [0.718, 0.825] | 0.667 | 0.929 |
| KEELE [capped n30/t10.0] | 0.692 [0.600, 0.763] | 0.615 | 0.790 |
| MDBStemSynth [capped n30/t10.0] | 0.876 [0.824, 0.919] | 0.917 | 0.838 |
| MIR1K [capped n30/t10.0] | 0.898 [0.879, 0.918] | 0.856 | 0.943 |
| MOCHA [capped n30/t10.0] | 0.744 [0.687, 0.795] | 0.671 | 0.834 |
| NSynth [capped n30/t10.0] | 0.864 [0.759, 0.947] | 0.819 | 0.913 |
| OSFGlottis (sparse) [capped n30/t10.0] | 0.772 [0.619, 0.856] | 0.891 | 0.681 |
| PTDB [capped n30/t10.0] | 0.776 [0.726, 0.821] | 0.750 | 0.804 |
| SVD [capped n30/t10.0] | 0.772 [0.748, 0.797] | 0.693 | 0.872 |
| SpeechSynth [capped n30/t10.0] | 0.924 [0.900, 0.944] | 0.907 | 0.942 |
| URMP [capped n30/t10.0] | 0.635 [0.542, 0.722] | 0.484 | 0.924 |
| Vocadito [capped n30/t10.0] | 0.885 [0.833, 0.914] | 0.881 | 0.888 |

Track score (equal per dataset): **0.791** [0.774, 0.804].

Where the recall comes from: pitch recall 0.795 = voicing recall 0.851 x the correct rate (exact+close, below).

### Praat

| **Dataset** | **pitch F1 [95% CI]** | **pitch recall** | **pitch precision** |
|---|---|---|---|
| APLAWD [capped n30/t10.0] | 0.754 [0.681, 0.791] | 0.668 | 0.864 |
| AVID (sparse) [capped n30/t10.0] | 0.496 [0.424, 0.573] | 0.798 | 0.360 |
| Bach10Synth [capped n30/t10.0] | 0.986 [0.984, 0.989] | 0.975 | 0.997 |
| CMUArctic [capped n30/t10.0] | 0.814 [0.771, 0.847] | 0.748 | 0.893 |
| FDA [capped n30/t10.0] | 0.786 [0.750, 0.816] | 0.683 | 0.925 |
| KEELE [capped n30/t10.0] | 0.769 [0.716, 0.810] | 0.677 | 0.889 |
| MDBStemSynth [capped n30/t10.0] | 0.965 [0.941, 0.982] | 0.946 | 0.986 |
| MIR1K [capped n30/t10.0] | 0.942 [0.924, 0.960] | 0.925 | 0.959 |
| MOCHA [capped n30/t10.0] | 0.719 [0.663, 0.769] | 0.639 | 0.824 |
| NSynth [capped n30/t10.0] | 0.881 [0.811, 0.934] | 0.818 | 0.954 |
| OSFGlottis (sparse) [capped n30/t10.0] | 0.732 [0.574, 0.831] | 0.898 | 0.617 |
| PTDB [capped n30/t10.0] | 0.748 [0.700, 0.790] | 0.663 | 0.858 |
| SVD [capped n30/t10.0] | 0.793 [0.768, 0.814] | 0.744 | 0.848 |
| SpeechSynth [capped n30/t10.0] | 0.947 [0.926, 0.961] | 0.913 | 0.984 |
| URMP [capped n30/t10.0] | 0.921 [0.900, 0.942] | 0.930 | 0.913 |
| Vocadito [capped n30/t10.0] | 0.928 [0.869, 0.957] | 0.937 | 0.919 |

Track score (equal per dataset): **0.824** [0.809, 0.835].

Where the recall comes from: pitch recall 0.881 = voicing recall 0.920 x the correct rate (exact+close, below).

### RAPT

| **Dataset** | **pitch F1 [95% CI]** | **pitch recall** | **pitch precision** |
|---|---|---|---|
| APLAWD [capped n30/t10.0] | 0.777 [0.727, 0.809] | 0.720 | 0.844 |
| AVID (sparse) [capped n30/t10.0] | 0.488 [0.393, 0.577] | 0.779 | 0.355 |
| Bach10Synth [capped n30/t10.0] | 0.987 [0.977, 0.993] | 0.983 | 0.990 |
| CMUArctic [capped n30/t10.0] | 0.787 [0.744, 0.840] | 0.763 | 0.812 |
| FDA [capped n30/t10.0] | 0.781 [0.769, 0.793] | 0.706 | 0.874 |
| KEELE [capped n30/t10.0] | 0.751 [0.704, 0.790] | 0.700 | 0.811 |
| MDBStemSynth [capped n30/t10.0] | 0.933 [0.889, 0.961] | 0.898 | 0.970 |
| MIR1K [capped n30/t10.0] | 0.932 [0.923, 0.943] | 0.926 | 0.937 |
| MOCHA [capped n30/t10.0] | 0.700 [0.644, 0.748] | 0.641 | 0.772 |
| NSynth [capped n30/t10.0] | 0.873 [0.741, 0.960] | 0.856 | 0.890 |
| OSFGlottis (sparse) [capped n30/t10.0] | 0.794 [0.679, 0.856] | 0.896 | 0.712 |
| PTDB [capped n30/t10.0] | 0.706 [0.663, 0.744] | 0.651 | 0.771 |
| SVD [capped n30/t10.0] | 0.785 [0.752, 0.814] | 0.755 | 0.817 |
| SpeechSynth [capped n30/t10.0] | 0.935 [0.907, 0.956] | 0.919 | 0.953 |
| URMP [capped n30/t10.0] | 0.904 [0.877, 0.929] | 0.893 | 0.915 |
| Vocadito [capped n30/t10.0] | 0.932 [0.916, 0.943] | 0.953 | 0.911 |

Track score (equal per dataset): **0.816** [0.801, 0.828].

Where the recall comes from: pitch recall 0.876 = voicing recall 0.935 x the correct rate (exact+close, below).

### REAPER

| **Dataset** | **pitch F1 [95% CI]** | **pitch recall** | **pitch precision** |
|---|---|---|---|
| APLAWD [capped n30/t10.0] | 0.789 [0.751, 0.830] | 0.707 | 0.891 |
| AVID (sparse) [capped n30/t10.0] | 0.477 [0.385, 0.568] | 0.779 | 0.343 |
| Bach10Synth [capped n30/t10.0] | 0.985 [0.979, 0.989] | 0.982 | 0.988 |
| KEELE [capped n30/t10.0] | 0.730 [0.678, 0.775] | 0.652 | 0.829 |
| MDBStemSynth [capped n30/t10.0] | 0.941 [0.899, 0.970] | 0.921 | 0.962 |
| MIR1K [capped n30/t10.0] | 0.932 [0.917, 0.944] | 0.914 | 0.950 |
| OSFGlottis (sparse) [capped n30/t10.0] | 0.726 [0.568, 0.828] | 0.883 | 0.616 |
| PTDB [capped n30/t10.0] | 0.723 [0.669, 0.768] | 0.672 | 0.782 |
| URMP [capped n30/t10.0] | 0.841 [0.772, 0.897] | 0.804 | 0.881 |
| Vocadito [capped n30/t10.0] | 0.916 [0.894, 0.933] | 0.916 | 0.917 |
| CMUArctic | 0.000 (crashed) | 0.000 | 0.000 |
| FDA | 0.000 (crashed) | 0.000 | 0.000 |
| MOCHA | 0.000 (crashed) | 0.000 | 0.000 |
| NSynth | 0.000 (crashed) | 0.000 | 0.000 |
| SVD | 0.000 (crashed) | 0.000 | 0.000 |
| SpeechSynth | 0.000 (crashed) | 0.000 | 0.000 |

Track score (equal per dataset): **0.504** [0.490, 0.514] over 10 completed + 6 crashed.

Where the recall comes from: pitch recall 0.879 = voicing recall 0.931 x the correct rate (exact+close, below).

### RMVPE

| **Dataset** | **pitch F1 [95% CI]** | **pitch recall** | **pitch precision** |
|---|---|---|---|
| APLAWD [capped n30/t10.0] | 0.846 [0.793, 0.872] | 0.766 | 0.944 |
| AVID (sparse) [capped n30/t10.0] | 0.531 [0.453, 0.609] | 0.845 | 0.387 |
| Bach10Synth [capped n30/t10.0] | 0.990 [0.987, 0.993] | 0.984 | 0.997 |
| CMUArctic [capped n30/t10.0] | 0.872 [0.844, 0.890] | 0.812 | 0.941 |
| FDA [capped n30/t10.0] | 0.862 [0.856, 0.868] | 0.797 | 0.938 |
| KEELE [capped n30/t10.0] | 0.822 [0.784, 0.851] | 0.758 | 0.898 |
| MDBStemSynth [capped n30/t10.0] | 0.904 [0.849, 0.945] | 0.881 | 0.928 |
| MIR1K [capped n30/t10.0] | 0.957 [0.951, 0.966] | 0.949 | 0.966 |
| MOCHA [capped n30/t10.0] | 0.772 [0.713, 0.830] | 0.673 | 0.903 |
| NSynth [capped n30/t10.0] | 0.903 [0.829, 0.951] | 0.836 | 0.982 |
| OSFGlottis (sparse) [capped n30/t10.0] | 0.754 [0.590, 0.857] | 0.930 | 0.634 |
| PTDB [capped n30/t10.0] | 0.796 [0.748, 0.836] | 0.733 | 0.871 |
| SVD [capped n30/t10.0] | 0.838 [0.817, 0.857] | 0.774 | 0.915 |
| SpeechSynth [capped n30/t10.0] | 0.972 [0.958, 0.982] | 0.977 | 0.966 |
| URMP [capped n30/t10.0] | 0.878 [0.822, 0.918] | 0.807 | 0.962 |
| Vocadito [capped n30/t10.0] | 0.956 [0.946, 0.964] | 0.973 | 0.940 |

Track score (equal per dataset): **0.853** [0.838, 0.864].

Where the recall comes from: pitch recall 0.880 = voicing recall 0.903 x the correct rate (exact+close, below).

### SPICE

| **Dataset** | **pitch F1 [95% CI]** | **pitch recall** | **pitch precision** |
|---|---|---|---|
| APLAWD [capped n30/t10.0] | 0.510 [0.442, 0.580] | 0.511 | 0.509 |
| AVID (sparse) [capped n30/t10.0] | 0.364 [0.315, 0.414] | 0.730 | 0.243 |
| Bach10Synth [capped n30/t10.0] | 0.910 [0.894, 0.926] | 0.859 | 0.968 |
| CMUArctic [capped n30/t10.0] | 0.596 [0.472, 0.700] | 0.669 | 0.537 |
| FDA [capped n30/t10.0] | 0.643 [0.538, 0.727] | 0.663 | 0.625 |
| KEELE [capped n30/t10.0] | 0.591 [0.513, 0.660] | 0.617 | 0.568 |
| MDBStemSynth [capped n30/t10.0] | 0.863 [0.810, 0.905] | 0.895 | 0.833 |
| MIR1K [capped n30/t10.0] | 0.908 [0.894, 0.924] | 0.953 | 0.867 |
| MOCHA [capped n30/t10.0] | 0.586 [0.521, 0.633] | 0.610 | 0.564 |
| NSynth [capped n30/t10.0] | 0.855 [0.769, 0.917] | 0.816 | 0.899 |
| OSFGlottis (sparse) [capped n30/t10.0] | 0.623 [0.480, 0.716] | 0.886 | 0.481 |
| PTDB [capped n30/t10.0] | 0.544 [0.512, 0.582] | 0.580 | 0.513 |
| SVD [capped n30/t10.0] | 0.670 [0.642, 0.697] | 0.705 | 0.639 |
| SpeechSynth [capped n30/t10.0] | 0.898 [0.871, 0.923] | 0.937 | 0.862 |
| URMP [capped n30/t10.0] | 0.862 [0.807, 0.907] | 0.808 | 0.924 |
| Vocadito [capped n30/t10.0] | 0.869 [0.841, 0.893] | 0.943 | 0.806 |

Track score (equal per dataset): **0.706** [0.687, 0.720].

Where the recall comes from: pitch recall 0.825 = voicing recall 0.899 x the correct rate (exact+close, below).

### SWIPE

| **Dataset** | **pitch F1 [95% CI]** | **pitch recall** | **pitch precision** |
|---|---|---|---|
| APLAWD [capped n30/t10.0] | 0.725 [0.666, 0.772] | 0.639 | 0.839 |
| AVID (sparse) [capped n30/t10.0] | 0.479 [0.419, 0.536] | 0.791 | 0.343 |
| CMUArctic [capped n30/t10.0] | 0.772 [0.732, 0.829] | 0.710 | 0.845 |
| FDA [capped n30/t10.0] | 0.763 [0.680, 0.827] | 0.680 | 0.869 |
| KEELE [capped n30/t10.0] | 0.732 [0.654, 0.794] | 0.636 | 0.862 |
| MDBStemSynth [capped n30/t10.0] | 0.955 [0.925, 0.977] | 0.949 | 0.962 |
| MOCHA [capped n30/t10.0] | 0.709 [0.653, 0.754] | 0.640 | 0.795 |
| NSynth [capped n30/t10.0] | 0.896 [0.836, 0.947] | 0.916 | 0.877 |
| OSFGlottis (sparse) [capped n30/t10.0] | 0.715 [0.557, 0.815] | 0.897 | 0.595 |
| PTDB [capped n30/t10.0] | 0.690 [0.648, 0.729] | 0.601 | 0.808 |
| SVD [capped n30/t10.0] | 0.782 [0.763, 0.801] | 0.738 | 0.833 |
| SpeechSynth [capped n30/t10.0] | 0.962 [0.949, 0.973] | 0.939 | 0.986 |
| URMP [capped n30/t10.0] | 0.889 [0.851, 0.922] | 0.907 | 0.872 |
| Vocadito [capped n30/t10.0] | 0.904 [0.880, 0.926] | 0.916 | 0.893 |
| Bach10Synth | 0.000 (crashed) | 0.000 | 0.000 |
| MIR1K | 0.000 (crashed) | 0.000 | 0.000 |

Track score (equal per dataset): **0.686** [0.669, 0.698] over 14 completed + 2 crashed.

Where the recall comes from: pitch recall 0.842 = voicing recall 0.890 x the correct rate (exact+close, below).

### SwiftF0

| **Dataset** | **pitch F1 [95% CI]** | **pitch recall** | **pitch precision** |
|---|---|---|---|
| APLAWD [capped n30/t10.0] | 0.586 [0.501, 0.627] | 0.559 | 0.616 |
| AVID (sparse) [capped n30/t10.0] | 0.442 [0.379, 0.510] | 0.689 | 0.325 |
| Bach10Synth [capped n30/t10.0] | 0.977 [0.971, 0.982] | 0.963 | 0.991 |
| CMUArctic [capped n30/t10.0] | 0.718 [0.676, 0.785] | 0.698 | 0.740 |
| FDA [capped n30/t10.0] | 0.703 [0.610, 0.794] | 0.656 | 0.759 |
| KEELE [capped n30/t10.0] | 0.673 [0.603, 0.743] | 0.638 | 0.713 |
| MDBStemSynth [capped n30/t10.0] | 0.888 [0.841, 0.927] | 0.882 | 0.893 |
| MIR1K [capped n30/t10.0] | 0.946 [0.934, 0.958] | 0.935 | 0.957 |
| MOCHA [capped n30/t10.0] | 0.676 [0.619, 0.715] | 0.604 | 0.768 |
| NSynth [capped n30/t10.0] | 0.934 [0.877, 0.972] | 0.949 | 0.920 |
| OSFGlottis (sparse) [capped n30/t10.0] | 0.747 [0.606, 0.828] | 0.885 | 0.646 |
| PTDB [capped n30/t10.0] | 0.622 [0.580, 0.667] | 0.587 | 0.662 |
| SVD [capped n30/t10.0] | 0.740 [0.712, 0.768] | 0.692 | 0.796 |
| SpeechSynth [capped n30/t10.0] | 0.934 [0.915, 0.950] | 0.925 | 0.944 |
| URMP [capped n30/t10.0] | 0.887 [0.851, 0.918] | 0.857 | 0.921 |
| Vocadito [capped n30/t10.0] | 0.912 [0.892, 0.927] | 0.942 | 0.883 |

Track score (equal per dataset): **0.774** [0.758, 0.787].

Where the recall comes from: pitch recall 0.853 = voicing recall 0.916 x the correct rate (exact+close, below).

### TorchCREPE

| **Dataset** | **pitch F1 [95% CI]** | **pitch recall** | **pitch precision** |
|---|---|---|---|
| APLAWD [capped n30/t10.0] | 0.759 [0.699, 0.796] | 0.632 | 0.948 |
| AVID (sparse) [capped n30/t10.0] | 0.538 [0.465, 0.607] | 0.746 | 0.421 |
| Bach10Synth [capped n30/t10.0] | 0.985 [0.982, 0.988] | 0.975 | 0.995 |
| CMUArctic [capped n30/t10.0] | 0.815 [0.751, 0.869] | 0.724 | 0.934 |
| FDA [capped n30/t10.0] | 0.806 [0.766, 0.840] | 0.732 | 0.898 |
| KEELE [capped n30/t10.0] | 0.775 [0.726, 0.813] | 0.685 | 0.891 |
| MDBStemSynth [capped n30/t10.0] | 0.821 [0.707, 0.900] | 0.699 | 0.995 |
| MIR1K [capped n30/t10.0] | 0.654 [0.499, 0.811] | 0.489 | 0.984 |
| MOCHA [capped n30/t10.0] | 0.725 [0.663, 0.778] | 0.623 | 0.868 |
| NSynth [capped n30/t10.0] | 0.922 [0.843, 0.970] | 0.894 | 0.952 |
| OSFGlottis (sparse) [capped n30/t10.0] | 0.760 [0.594, 0.863] | 0.900 | 0.657 |
| PTDB [capped n30/t10.0] | 0.723 [0.689, 0.754] | 0.640 | 0.831 |
| SVD [capped n30/t10.0] | 0.804 [0.784, 0.823] | 0.713 | 0.921 |
| SpeechSynth [capped n30/t10.0] | 0.964 [0.955, 0.973] | 0.950 | 0.979 |
| URMP [capped n30/t10.0] | 0.835 [0.750, 0.898] | 0.732 | 0.973 |
| Vocadito [capped n30/t10.0] | 0.937 [0.922, 0.948] | 0.958 | 0.916 |

Track score (equal per dataset): **0.801** [0.781, 0.818].

Where the recall comes from: pitch recall 0.778 = voicing recall 0.798 x the correct rate (exact+close, below).

### YAAPT

| **Dataset** | **pitch F1 [95% CI]** | **pitch recall** | **pitch precision** |
|---|---|---|---|
| APLAWD [capped n30/t10.0] | 0.703 [0.628, 0.756] | 0.641 | 0.779 |
| AVID (sparse) [capped n30/t10.0] | 0.406 [0.343, 0.467] | 0.655 | 0.295 |
| Bach10Synth [capped n30/t10.0] | 0.737 [0.682, 0.784] | 0.722 | 0.754 |
| CMUArctic [capped n30/t10.0] | 0.744 [0.711, 0.773] | 0.724 | 0.766 |
| FDA [capped n30/t10.0] | 0.699 [0.698, 0.701] | 0.685 | 0.714 |
| KEELE [capped n30/t10.0] | 0.675 [0.639, 0.707] | 0.656 | 0.696 |
| MDBStemSynth [capped n30/t10.0] | 0.742 [0.628, 0.833] | 0.743 | 0.741 |
| MIR1K [capped n30/t10.0] | 0.864 [0.837, 0.888] | 0.855 | 0.873 |
| MOCHA [capped n30/t10.0] | 0.600 [0.532, 0.661] | 0.567 | 0.637 |
| NSynth [capped n30/t10.0] | 0.666 [0.441, 0.865] | 0.637 | 0.698 |
| OSFGlottis (sparse) [capped n30/t10.0] | 0.693 [0.523, 0.803] | 0.869 | 0.577 |
| PTDB [capped n30/t10.0] | 0.622 [0.584, 0.660] | 0.619 | 0.625 |
| SVD [capped n30/t10.0] | 0.754 [0.727, 0.779] | 0.715 | 0.798 |
| SpeechSynth [capped n30/t10.0] | 0.853 [0.819, 0.882] | 0.857 | 0.849 |
| URMP [capped n30/t10.0] | 0.629 [0.516, 0.727] | 0.624 | 0.634 |
| Vocadito [capped n30/t10.0] | 0.886 [0.868, 0.902] | 0.914 | 0.860 |

Track score (equal per dataset): **0.705** [0.682, 0.724].

Where the recall comes from: pitch recall 0.734 = voicing recall 0.940 x the correct rate (exact+close, below).

### pYIN

| **Dataset** | **pitch F1 [95% CI]** | **pitch recall** | **pitch precision** |
|---|---|---|---|
| APLAWD [capped n30/t10.0] | 0.312 [0.204, 0.361] | 0.205 | 0.654 |
| AVID (sparse) [capped n30/t10.0] | 0.304 [0.242, 0.371] | 0.503 | 0.218 |
| Bach10Synth [capped n30/t10.0] | 0.961 [0.956, 0.965] | 0.947 | 0.975 |
| CMUArctic [capped n30/t10.0] | 0.514 [0.320, 0.670] | 0.419 | 0.666 |
| FDA [capped n30/t10.0] | 0.458 [0.287, 0.571] | 0.349 | 0.665 |
| KEELE [capped n30/t10.0] | 0.433 [0.346, 0.509] | 0.326 | 0.646 |
| MDBStemSynth [capped n30/t10.0] | 0.840 [0.781, 0.891] | 0.809 | 0.875 |
| MIR1K [capped n30/t10.0] | 0.865 [0.839, 0.887] | 0.859 | 0.872 |
| MOCHA [capped n30/t10.0] | 0.490 [0.437, 0.543] | 0.416 | 0.596 |
| NSynth [capped n30/t10.0] | 0.918 [0.860, 0.958] | 0.927 | 0.908 |
| OSFGlottis (sparse) [capped n30/t10.0] | 0.650 [0.512, 0.733] | 0.814 | 0.541 |
| PTDB [capped n30/t10.0] | 0.414 [0.352, 0.469] | 0.319 | 0.593 |
| SVD [capped n30/t10.0] | 0.547 [0.506, 0.586] | 0.492 | 0.615 |
| SpeechSynth [capped n30/t10.0] | 0.710 [0.657, 0.759] | 0.648 | 0.785 |
| URMP [capped n30/t10.0] | 0.911 [0.887, 0.934] | 0.902 | 0.919 |
| Vocadito [capped n30/t10.0] | 0.824 [0.793, 0.848] | 0.823 | 0.824 |

Track score (equal per dataset): **0.635** [0.609, 0.652].

Where the recall comes from: pitch recall 0.757 = voicing recall 0.834 x the correct rate (exact+close, below).


### Error breakdown

On frames scored for pitch, at theta\*, pooled over all conditions. The four classes are disjoint and cover every scored frame, so a row reads as a sentence: *exact on X, loses tuning on Y, wanders on Z, picks the wrong pitch on W*. `cents bias` is the mean SIGNED error over the in-tolerance frames only (so its denominator differs from MAE's): 0 means symmetric scatter, non-zero means the tracker sits systematically sharp (+) or flat (-) and can be corrected by subtracting it. It is the only signed column; every other one is blind to the direction of the error.

| **Algorithm** | **exact** (<10c) | **close** (10-50c) | **off** (50-200c) | **wrong** (>200c) | **cents MAE** | **cents bias** |
|---|---|---|---|---|---|---|
| BasicPitch | 0.339 | 0.531 | 0.093 | 0.038 | 74.2 | -4.4 |
| CREPE | 0.724 | 0.213 | 0.027 | 0.036 | 61.9 | +0.0 |
| DIO | 0.611 | 0.249 | 0.070 | 0.071 | 108.5 | -0.5 |
| Harvest | 0.655 | 0.235 | 0.067 | 0.043 | 62.9 | -0.4 |
| PENN | 0.584 | 0.316 | 0.057 | 0.044 | 59.8 | -2.4 |
| Praat | 0.742 | 0.187 | 0.035 | 0.036 | 62.6 | +0.4 |
| RAPT | 0.679 | 0.229 | 0.056 | 0.036 | 54.6 | -0.2 |
| REAPER | 0.576 | 0.316 | 0.037 | 0.071 | 106.6 | -0.2 |
| RMVPE | 0.756 | 0.203 | 0.024 | 0.017 | 28.5 | -0.8 |
| SPICE | 0.584 | 0.333 | 0.065 | 0.019 | 34.0 | -0.5 |
| SWIPE | 0.583 | 0.306 | 0.056 | 0.055 | 117.1 | +3.5 |
| SwiftF0 | 0.481 | 0.438 | 0.068 | 0.013 | 32.6 | -0.4 |
| TorchCREPE | 0.515 | 0.446 | 0.029 | 0.009 | 23.3 | -0.0 |
| YAAPT | 0.102 | 0.549 | 0.122 | 0.226 | 329.5 | -19.9 |
| pYIN | 0.620 | 0.292 | 0.065 | 0.023 | 39.5 | +1.2 |

### By pitch band

Correct rate (exact+close) with the frame count it rests on, by ground-truth band. A register collapse shows up here and nowhere else in the aggregate; a small count means the cell is not evidence.

| **Algorithm** | **bass** (<80 Hz) | **low** (80-260 Hz) | **mid** (260-650 Hz) | **high** (650-1050 Hz) | **vhigh** (>=1050 Hz) |
|---|---|---|---|---|---|
| BasicPitch | 0.85 (27,166) | 0.83 (532,501) | 0.92 (339,496) | 0.92 (19,766) | 0.93 (3,948) |
| CREPE | 0.91 (20,670) | 0.92 (561,350) | 0.97 (335,292) | 0.94 (16,946) | 0.98 (4,292) |
| DIO | 0.80 (21,697) | 0.82 (437,484) | 0.92 (257,202) | 0.97 (12,896) | 0.97 (3,801) |
| Harvest | 0.84 (26,569) | 0.87 (576,813) | 0.93 (289,205) | 0.92 (14,605) | 0.95 (3,422) |
| PENN | 0.85 (22,822) | 0.88 (395,429) | 0.94 (194,045) | 0.95 (8,378) | 0.72 (1,895) |
| Praat | 0.94 (20,470) | 0.92 (437,827) | 0.94 (265,414) | 0.88 (15,378) | 0.93 (4,119) |
| RAPT | 0.88 (12,256) | 0.91 (380,192) | 0.93 (230,281) | 0.82 (12,602) | 0.29 (3,171) |
| REAPER | 0.90 (14,575) | 0.91 (347,749) | 0.89 (248,481) | 0.60 (10,133) | 0.00 (607) |
| RMVPE | 0.94 (15,913) | 0.95 (624,313) | 0.98 (364,128) | 0.96 (17,583) | 0.62 (285) |
| SPICE | 0.83 (16,003) | 0.89 (441,838) | 0.96 (300,279) | 0.97 (11,654) | 0.07 (14) |
| SWIPE | 0.87 (10,146) | 0.87 (316,528) | 0.93 (168,365) | 0.92 (9,482) | 0.57 (2,875) |
| SwiftF0 | 0.84 (22,796) | 0.90 (479,677) | 0.96 (279,389) | 0.93 (10,915) | 1.00 (4,953) |
| TorchCREPE | 0.98 (11,066) | 0.95 (440,253) | 0.98 (251,351) | 0.99 (12,721) | 1.00 (3,806) |
| YAAPT | 0.67 (27,228) | 0.75 (622,946) | 0.52 (342,749) | 0.00 (17,971) | 0.00 (5,715) |
| pYIN | 0.94 (15,988) | 0.89 (351,439) | 0.94 (240,938) | 0.95 (14,230) | 1.00 (4,115) |

### Statistical ties

Per clean dataset: the best pitch F1, and every algorithm whose paired 95% CI of the difference over the clips both scored includes 0. A tied algorithm is not beaten on that dataset, whatever the third decimal suggests. Many pairs are compared with no multiple-comparisons correction, so read a single tie as descriptive.

| **Dataset** | **best pitch F1** | **tied with best** |
|---|---|---|
| APLAWD | RMVPE (0.846) | none |
| AVID | PENN (0.562) | RMVPE |
| Bach10Synth | Harvest (0.998) | none |
| CMUArctic | RMVPE (0.872) | none |
| FDA | RMVPE (0.862) | none |
| KEELE | RMVPE (0.822) | none |
| MDBStemSynth | Praat (0.965) | Harvest, REAPER, SWIPE |
| MIR1K | RMVPE (0.957) | none |
| MOCHA | RMVPE (0.772) | none |
| NSynth | BasicPitch (0.942) | CREPE, DIO, PENN, RAPT, RMVPE, SWIPE, SwiftF0, TorchCREPE, pYIN |
| OSFGlottis | RAPT (0.794) | PENN, RMVPE, TorchCREPE |
| PTDB | RMVPE (0.796) | none |
| SVD | RMVPE (0.838) | none |
| SpeechSynth | RMVPE (0.972) | TorchCREPE |
| URMP | BasicPitch (0.925) | CREPE, Praat, RAPT, pYIN |
| Vocadito | RMVPE (0.956) | none |


### Correctness by domain

Clean pitch F1, equal weight per dataset within each domain.

| **Algorithm** | **Speech** | **Music** |
|---|---|---|
| BasicPitch | 0.540 | 0.875 |
| CREPE | 0.757 | 0.939 |
| DIO | 0.744 | 0.904 |
| Harvest | 0.647 | 0.915 |
| PENN | 0.753 | 0.854 |
| Praat | 0.756 | 0.937 |
| RAPT | 0.750 | 0.927 |
| REAPER | 0.689 | 0.923 |
| RMVPE | 0.806 | 0.931 |
| SPICE | 0.603 | 0.878 |
| SWIPE | 0.733 | 0.911 |
| SwiftF0 | 0.684 | 0.924 |
| TorchCREPE | 0.767 | 0.859 |
| YAAPT | 0.675 | 0.754 |
| pYIN | 0.483 | 0.886 |

## Track 2: Noise

Absolute pitch F1 on the degraded conditions at theta\*, equal weight per dataset. This is how well the tracker works in noise, not how much of its clean performance it kept: a retention ratio rises when clean performance falls, so it rewards being bad on clean and is not scored.

| **Algorithm** | **Score** | **#conditions** | **drop pp [95% CI]** | **chime** | **codec** | **demand** | **fade** | **gain** | **pink** | **pink_snr+10** | **pink_snr-5** | **reverb** | **room** | **telephone** | **white** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BasicPitch | 0.531 [0.523, 0.534] | 12 | 13.2 to 13.9 | 0.476 | 0.665 | 0.471 | 0.643 | 0.655 | 0.486 | 0.640 | 0.286 | 0.542 | 0.530 | 0.400 | 0.581 |
| CREPE | 0.637 [0.629, 0.640] | 12 | 18.5 to 19.4 | 0.541 | 0.824 | 0.534 | 0.823 | 0.825 | 0.539 | 0.783 | 0.217 | 0.611 | 0.597 | 0.652 | 0.697 |
| DIO | 0.443 [0.435, 0.447] | 12 | 35.6 to 36.8 | 0.277 | 0.804 | 0.160 | 0.804 | 0.804 | 0.065 | 0.514 | 0.006 | 0.546 | 0.512 | 0.373 | 0.453 |
| Harvest | 0.542 [0.534, 0.547] | 12 | 20.0 to 21.2 | 0.472 | 0.746 | 0.407 | 0.749 | 0.747 | 0.344 | 0.689 | 0.102 | 0.546 | 0.521 | 0.603 | 0.581 |
| PENN | 0.426 [0.418, 0.430] | 12 | 35.9 to 37.2 | 0.286 | 0.791 | 0.139 | 0.748 | 0.500 | 0.048 | 0.444 | 0.006 | 0.572 | 0.511 | 0.608 | 0.456 |
| Praat | 0.492 [0.487, 0.495] | 12 | 32.8 to 33.5 | 0.271 | 0.825 | 0.276 | 0.667 | 0.824 | 0.179 | 0.765 | 0.011 | 0.606 | 0.590 | 0.776 | 0.116 |
| RAPT | 0.400 [0.395, 0.403] | 12 | 41.2 to 42.1 | 0.213 | 0.817 | 0.185 | 0.682 | 0.057 | 0.095 | 0.728 | 0.002 | 0.599 | 0.580 | 0.789 | 0.054 |
| REAPER | 0.295 [0.291, 0.298] | 12 | 30.0 to 31.1 | 0.152 | 0.503 | 0.139 | 0.417 | 0.468 | 0.089 | 0.366 | 0.009 | 0.345 | 0.332 | 0.492 | 0.225 |
| RMVPE | 0.756 [0.750, 0.759] | 12 | 9.4 to 10.1 | 0.702 | 0.853 | 0.754 | 0.823 | 0.780 | 0.793 | 0.843 | 0.669 | 0.646 | 0.612 | 0.763 | 0.837 |
| SPICE | 0.504 [0.494, 0.508] | 12 | 19.8 to 20.8 | 0.422 | 0.706 | 0.359 | 0.700 | 0.707 | 0.366 | 0.696 | 0.066 | 0.554 | 0.539 | 0.349 | 0.583 |
| SWIPE | 0.363 [0.355, 0.367] | 12 | 38.0 to 39.3 | 0.199 | 0.632 | 0.073 | 0.678 | 0.700 | 0.030 | 0.556 | 0.002 | 0.464 | 0.398 | 0.422 | 0.206 |
| SwiftF0 | 0.525 [0.517, 0.528] | 12 | 24.5 to 25.5 | 0.424 | 0.775 | 0.348 | 0.736 | 0.673 | 0.240 | 0.712 | 0.042 | 0.566 | 0.551 | 0.600 | 0.629 |
| TorchCREPE | 0.583 [0.574, 0.587] | 12 | 21.4 to 22.4 | 0.435 | 0.803 | 0.406 | 0.801 | 0.801 | 0.425 | 0.720 | 0.186 | 0.596 | 0.579 | 0.657 | 0.589 |
| YAAPT | 0.496 [0.489, 0.502] | 12 | 20.3 to 21.4 | 0.359 | 0.703 | 0.348 | 0.627 | 0.698 | 0.297 | 0.640 | 0.062 | 0.518 | 0.518 | 0.666 | 0.520 |
| pYIN | 0.347 [0.340, 0.351] | 12 | 28.2 to 29.1 | 0.080 | 0.634 | 0.040 | 0.633 | 0.635 | 0.013 | 0.511 | 0.000 | 0.546 | 0.521 | 0.549 | 0.008 |

`drop` is the paired clean-minus-degraded difference in pitch F1, in percentage points, resampled over the clips both sides scored. `#conditions` is the denominator of the score: two algorithms averaged over different numbers of conditions are not directly comparable.

## Track 3: Signal types

Synthetic families with exact labels. Per family: unconditional pitch recall at theta\*, so a family the tracker refuses to voice scores low rather than vanishing. Score = mean over families, each family one equally-weighted probe question. A dead family is one scoring exactly 0 -- the capability is absent, not merely weak.

| **Algorithm** | **Score** | **worst family** | **dead** | **which** |
|---|---|---|---|---|
| BasicPitch | 0.809 | missing_f0 (0.00) | 2 | missing_f0, unresolved |
| CREPE | 0.893 | tilt_low (0.00) | 1 | tilt_low |
| DIO | 0.685 | harm_vhigh (0.00) | 6 | harm_vhigh, interference, missing_f0, sine_vhigh, tilt_vhigh, unresolved |
| Harvest | 0.510 | harm_high (0.00) | 6 | harm_high, missing_f0, sine_level, sine_mid, tilt_high, unresolved |
| PENN | 0.675 | harm_vhigh (0.00) | 3 | harm_vhigh, sine_vhigh, tilt_vhigh |
| Praat | 0.953 | interference (0.65) | 0 | none |
| RAPT | 0.664 | harm_vhigh (0.00) | 3 | harm_vhigh, sine_vhigh, tilt_vhigh |
| REAPER | 0.222 | sine_low (0.00) | 17 | glide, harm_high, harm_low, harm_mid, harm_vhigh, irn, missing_f0, sine_bass, sine_high, sine_level, sine_low, sine_mid, sine_vhigh, tilt_high, tilt_mid, tilt_vhigh, unresolved |
| RMVPE | 0.471 | harm_vhigh (0.00) | 5 | harm_vhigh, sine_bass, sine_vhigh, tilt_vhigh, unresolved |
| SPICE | 0.507 | harm_vhigh (0.00) | 5 | harm_vhigh, missing_f0, sine_vhigh, tilt_vhigh, unresolved |
| SWIPE | 0.809 | tilt_low (0.00) | 2 | tilt_low, unresolved |
| SwiftF0 | 0.817 | sine_bass (0.00) | 2 | sine_bass, unresolved |
| TorchCREPE | 0.892 | tilt_low (0.00) | 1 | tilt_low |
| YAAPT | 0.399 | harm_high (0.00) | 11 | harm_high, harm_mid, harm_vhigh, interference, sine_high, sine_low, sine_mid, sine_vhigh, tilt_high, tilt_vhigh, vibrato_fast |
| pYIN | 0.776 | tilt_vhigh (0.00) | 1 | tilt_vhigh |

Controls (broadband, silence, whisper) contain no pitch, so the only readout is the false-positive rate (diagnostic, never scored): BasicPitch silence 0.052; DIO whisper 0.024; Harvest whisper 0.008; PENN whisper 0.004; REAPER whisper 0.148; RMVPE whisper 0.044; SPICE whisper 0.008; SWIPE whisper 0.208; YAAPT broadband 0.716; YAAPT silence 0.620; YAAPT whisper 0.256.

## Track 4: Tracking

Trajectory families with exact labels, read at theta\*. Steady tones score sigma0/(sigma0+error) on the TOTAL error hypot(jitter, bias) with sigma0=10 cents, so a constant output that is an octave off cannot win by having no jitter. Vibrato scores depth ratio x voiced fraction. Track score = mean of the two capabilities.

| **Algorithm** | **Score** | **steady_harm** | **steady_sine** | **vib_half** | **vib_one** | **vib_two** |
|---|---|---|---|---|---|---|
| BasicPitch | 0.833 | jitter 0.0c, bias +0.0c | jitter 0.0c, bias +0.0c | depth 0.00 x voiced 1.00 | depth 1.22 x voiced 1.00 | depth 1.19 x voiced 1.00 |
| CREPE | 0.914 | jitter 0.6c, bias +0.2c | jitter 0.8c, bias -0.6c | depth 0.91 x voiced 1.00 | depth 0.97 x voiced 1.00 | depth 0.95 x voiced 0.88 |
| DIO | 0.975 | jitter 0.2c, bias +0.0c | jitter 0.1c, bias -0.8c | depth 1.00 x voiced 1.00 | depth 1.00 x voiced 1.00 | depth 1.00 x voiced 1.00 |
| Harvest | 0.747 | jitter 0.0c, bias -0.0c | n/a | depth 0.99 x voiced 1.00 | depth 0.99 x voiced 1.00 | depth 0.99 x voiced 1.00 |
| PENN | 0.870 | jitter 0.9c, bias -1.6c | jitter 0.9c, bias -2.2c | depth 0.94 x voiced 1.00 | depth 0.90 x voiced 1.00 | depth 0.91 x voiced 1.00 |
| Praat | 0.987 | jitter 0.0c, bias +0.0c | jitter 0.0c, bias -0.0c | depth 0.96 x voiced 1.00 | depth 0.97 x voiced 1.00 | depth 0.99 x voiced 1.00 |
| RAPT | 0.953 | jitter 0.6c, bias +0.3c | jitter 0.3c, bias +0.0c | depth 0.95 x voiced 1.00 | depth 0.95 x voiced 1.00 | depth 0.95 x voiced 1.00 |
| REAPER | 0.317 | n/a | n/a | n/a | depth 0.95 x voiced 1.00 | depth 0.95 x voiced 1.00 |
| RMVPE | 0.940 | jitter 0.7c, bias -0.8c | jitter 0.8c, bias -0.3c | depth 0.98 x voiced 1.00 | depth 0.98 x voiced 1.00 | depth 0.94 x voiced 1.00 |
| SPICE | 0.729 | jitter 0.1c, bias -2.7c | jitter 0.0c, bias -4.4c | depth 0.64 x voiced 1.00 | depth 0.72 x voiced 1.00 | depth 0.79 x voiced 1.00 |
| SWIPE | 0.785 | jitter 0.0c, bias +0.6c | jitter 0.0c, bias +25.1c | depth 0.94 x voiced 1.00 | depth 0.96 x voiced 1.00 | depth 0.97 x voiced 1.00 |
| SwiftF0 | 0.881 | jitter 0.0c, bias +0.5c | jitter 0.1c, bias +1.2c | depth 0.77 x voiced 1.00 | depth 0.87 x voiced 1.00 | depth 0.89 x voiced 1.00 |
| TorchCREPE | 0.544 | jitter 8.0c, bias +6.8c | jitter 8.3c, bias +5.7c | depth 0.85 x voiced 1.00 | depth 0.94 x voiced 1.00 | depth 0.00 x voiced 0.00 |
| YAAPT | 0.192 | jitter 0.0c, bias -650.9c | jitter 79.3c, bias -972.6c | depth 0.51 x voiced 1.00 | depth 0.55 x voiced 1.00 | depth 0.05 x voiced 1.00 |
| pYIN | 0.506 | jitter 0.0c, bias +1.7c | jitter 0.0c, bias +5.0c | depth 0.45 x voiced 1.00 | depth 0.47 x voiced 0.63 | depth 0.00 x voiced 0.00 |

## Track 5: Notes

Not measured in this run.

## Track 6: Speed

Real-time factor on cpu under controlled conditions (serial, isolated, repeated): seconds of compute per second of audio, so lower is faster. Score = 1/(1+RTF).

| **Algorithm** | **RTF (cpu)** | **Score** |
|---|---|---|
| BasicPitch | 0.0308 | 0.970 |
| CREPE | 0.3924 | 0.718 |
| DIO | 0.0045 | 0.996 |
| Harvest | 0.0354 | 0.966 |
| PENN | 0.5862 | 0.630 |
| Praat | 0.0016 | 0.998 |
| RAPT | 0.0019 | 0.998 |
| REAPER | n/a | 0.000 |
| RMVPE | 0.1108 | 0.900 |
| SPICE | 0.0469 | 0.955 |
| SWIPE | 0.0185 | 0.982 |
| SwiftF0 | 0.0469 | 0.955 |
| TorchCREPE | 2.3857 | 0.295 |
| YAAPT | 0.0207 | 0.980 |
| pYIN | 0.1633 | 0.860 |

## Reliability

Nothing here is scored -- these are facts about running the code. A crashed cell still counts against the tracker's scores (it contributes 0 to its track), so `completed/attempted` explains a low score rather than excusing it. Speed is measured separately, and comparably, by the Speed track.

| **Algorithm** | **completed/attempted** |
|---|---|
| BasicPitch | 239/239 |
| CREPE | 239/239 |
| DIO | 239/239 |
| Harvest | 239/239 |
| PENN | 239/239 |
| Praat | 239/239 |
| RAPT | 239/239 |
| REAPER | 131/239 (4x IndexError, 79x exit -11, 25x exit -6) |
| RMVPE | 239/239 |
| SPICE | 239/239 |
| SWIPE | 219/239 (20x exit -11) |
| SwiftF0 | 239/239 |
| TorchCREPE | 239/239 |
| YAAPT | 239/239 |
| pYIN | 239/239 |

## Datasets

| **Dataset** | **Domain** | **Clips** | **Hours** | **Voiced %** | **f0 p5-p50-p95 (Hz)** | **f0 by band** |
|---|---|---|---|---|---|---|
| NSynth | Music | 3319 | 3.7 | 53 | 73-277-1480 | bass 8%, low 39%, mid 29%, high 12%, vhigh 12% |
| PTDB | Speech | 4718 | 9.6 | 26 | 84-156-236 | bass 3%, low 95%, mid 2% |
| MIR1K | Music | 1000 | 2.2 | 70 | 121-231-390 | low 63%, mid 37% |
| MDBStemSynth | Music | 230 | 15.6 | 40 | 73-209-704 | bass 11%, low 50%, mid 32%, high 6% |
| Vocadito | Music | 40 | 0.2 | 66 | 113-219-364 | low 69%, mid 31% |
| Bach10Synth | Music | 40 | 0.4 | 92 | 111-296-579 | low 38%, mid 59%, high 2% |
| SpeechSynth | Speech | 219 | 0.1 | 47 | 126-213-273 | low 90%, mid 10% |
| KEELE | Speech | 10 | 0.1 | 54 | 81-173-295 | bass 5%, low 85%, mid 10% |
| FDA | Speech | 100 | 0.1 | 42 | 91-205-296 | low 80%, mid 20% |
| MOCHA | Speech | 3690 | 4.3 | 46 | 98-188-355 | low 82%, mid 17% |
| CMUArctic | Speech | 3377 | 2.8 | 60 | 93-130-195 | bass 1%, low 99% |
| SVD | Speech | 634 | 0.3 | 73 | 87-181-287 | bass 3%, low 86%, mid 12% |
| APLAWD | Speech | 10979 | 2.7 | 58 | 81-134-251 | bass 4%, low 91%, mid 4% |
| OSFGlottis (sparse-voiced) | Speech | 14 | 6.9 | 20 | 113-188-257 | low 96%, mid 4% |
| AVID (sparse-voiced) | Speech | 51 | 18.3 | 21 | 114-208-346 | low 76%, mid 24% |
| M4Singer | Music | 20896 | 29.7 | 86 | 92-233-523 | bass 1%, low 55%, mid 43% |
| URMP | Music | 149 | 4.6 | 76 | 109-348-856 | bass 1%, low 31%, mid 54%, high 12%, vhigh 2% |

Bands: bass (<80 Hz), low (80-260 Hz), mid (260-650 Hz), high (650-1050 Hz), vhigh (>=1050 Hz). f0 statistics are over in-window voiced frames.

## Caveats

- **Sparse-voiced corpora** (OSFGlottis, AVID): mostly-unvoiced sessions, so precision
  denominators are dominated by silence false positives. Flagged in the tables, never
  dropped.
- **Score-grade pitch ground truth** (M4Singer): the notated note, not the performed f0.
  Its voicing labels are sound, so the corpus is kept for voicing, but it is excluded
  from every pitch number including theta\* selection.
- **Training-data leakage**: learned trackers may have trained on these public corpora.
  A clean-only advantage that collapses under degradation is a leakage signature; the
  degraded and synthetic tracks move inputs away from anything seen verbatim.
