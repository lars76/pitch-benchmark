# Metrics

Every definition the benchmark uses, stated once. The report links here rather than
repeating them.

## The contract

An algorithm is `(f0(t), q(t))`: a pitch estimate and a voicing confidence per frame.

`f0 <= 0` or `NaN` is a **voicing claim, never a pitch estimate**. At threshold theta a
frame counts as *voiced-with-pitch* iff `q >= theta` and `f0` is finite and positive.
Abstaining on a truly voiced frame costs recall; it never fabricates a cents error. A
tracker cannot buy accuracy by declining the hard frames.

## The one metric family

Let `n_correct` = frames that are truly voiced, given a pitch by the tracker, and within
50 cents. Let `tp / fp / fn` be the voicing confusion counts.

```
pitch recall    = n_correct / (tp + fn)      of the pitch that exists, how much it got
pitch precision = n_correct / (tp + fp)      of what it output, how much was right
pitch F1        = 2 n_correct / ((tp + fp) + (tp + fn))
```

Pure counts, no weights. Precision and recall share a numerator, so F1 is the *derived*
harmonic mean (a Dice overlap), not a chosen one.

This is an F1 score in the strict sense. Define the event *"the tracker produced a correct
pitch at this frame"*; then `TP = n_correct`, `FP = (tp+fp) - n_correct` (it output
something that was not right), `FN = (tp+fn) - n_correct` (pitch existed and was not
correctly covered), and the textbook `2TP / (2TP + FP + FN)` reproduces the formula above
exactly. The only unusual part is that a correct detection requires *two* conditions --
firing in the right place AND being close enough -- which is the same construction object
detection uses for F1 at an IoU threshold, and `mir_eval.transcription` uses for notes.

F1 ignores true negatives, i.e. correctly-called silence. That is deliberate: silence is
abundant here, and counting it would let a tracker that outputs almost nothing score well
on a mostly-unvoiced corpus. It is also why these numbers are NOT comparable to the
melody-extraction literature's Overall Accuracy, which does count them.

**Voicing precision / recall / F1 are the same three formulas with the pitch test
removed** — a tolerance wide enough to accept any pitch. That bound is finite, not a
limit: the largest possible error is `1200 log2(fmax/fmin)`, about 6600 cents over a
46–2094 Hz range. So the benchmark has one metric family, not two, and voicing F1 is its
endpoint rather than a separate measure.

## How this maps to the literature

The *structure* above is standard; the *name* "pitch F1" is ours. Frame-level pitch
tracking has its own established vocabulary, and a reader coming from it needs the bridge:

| literature metric | what it is | here |
|---|---|---|
| **RPA** (raw pitch accuracy) | of truly-voiced frames, the fraction within 50 cents | `correct_rate` = exact + close |
| **RCA** (raw chroma accuracy) | RPA ignoring octave errors | dropped: measured r = +0.97 with RPA |
| **VR / VFA** | voicing recall / false-alarm rate | voicing recall / precision |
| **OA** (overall accuracy) | correct voiced + correctly-called unvoiced, over all frames | not reported (see above) |
| **pitch F1** | — | the headline |

So **pitch F1 is not directly comparable to a published pitch-tracking number.** RPA is,
and is reported in the error breakdown. Compare against papers there, not on the
leaderboard.

## The operating point (theta\*)

One global **theta\*** per algorithm: the argmax over the threshold grid of the
equal-per-dataset mean pitch F1 on clean cells. Full (uncapped) cells are preferred; if
only capped baselines exist the fallback is used and stamped in the provenance. Ties
resolve to the lowest threshold.

It is frozen, and every track reads it. There is no per-cell threshold selection anywhere,
so no tracker is ever scored at an operating point chosen with knowledge of the test it is
being scored on.

Datasets excluded from pitch scoring (below) are excluded from theta\* selection too: a
threshold tuned partly against an annotation artifact would then govern every table.

## Error classes

All on `abs_cents = |1200 log2(pred/true)|` over scored frames. The four classes are
disjoint half-open intervals covering every scored frame, so they sum to 1 and only three
are independent.

| class | error size | what the user gets |
|---|---|---|
| **exact** | < 10 cents (0.6%) | indistinguishable from the truth |
| **close** | 10–50 cents (0.6–2.9%) | right note, imperfect tuning |
| **off** | 50–200 cents (2.9–12%) | audibly flat or sharp |
| **wrong** | > 200 cents (>12%) | a different pitch entirely |

`exact + close` is the literature's RPA. Cent thresholds are 12-TET conventions (50c = half
a semitone, the melody-extraction standard); percentages are given for readers outside
music.

**`exact` is the fine-precision column.** pitch F1 scores a frame 5 cents out and one 45
cents out identically, because both are inside the tolerance; `exact` is what separates
them. If tuning accuracy is what you care about -- tone languages, intonation research,
instrument tuning, following a synthesised sweep -- read `exact` and `cents MAE`, not the
headline.

**cents MAE** is the mean absolute error over scored frames, reported for literature
comparability. It is tail-dominated, so it measures wild errors more than typical ones.

**cents bias** is the mean *signed* error over the **in-tolerance frames only** (|error| <
50 cents), so its denominator is the correct-frame count, not the scored-frame count — it
is not comparable to cents MAE. It answers one question nothing else in the benchmark can:
**does the tracker sit systematically sharp (+) or flat (−)?** Every other column is a
function of |error| and is therefore sign-blind; a tracker consistently 30 cents sharp and
one scattering ±30 cents symmetrically are identical on pitch F1, on all four error
classes, and on cents MAE.

Its optimum is two-sided and exact: **0**. A non-zero value is the one defect in this
benchmark that is directly *correctable* — subtract it from every prediction. Typical
causes are a resampling or reference-frequency error, a mis-set A4, a training-label
offset, or (see below) a residual timestamp misalignment.

The in-tolerance restriction is what makes it mean that. Summed over all scored frames, the
mean signed error is dominated by the octave tail: it reported a 125-cent "offset" for a
tracker whose typical error was 5 cents, and subtracting that offset destroyed its accuracy
(exact rate 0.56 → 0.003). Restricting to frames already within tolerance is immune to the
tail and stays additive, which a median — the other obvious fix — would not.

**It assumes calibrated timestamps.** A timestamp offset produces a cents error
proportional to the local pitch slope, so on speech (which declines on average) it leaks
into this column. Every wrapper's timestamps are measured to within 2 ms, which bounds the
leak to about 1 cent; an uncorrected half-hop (8 ms) error would fake roughly 4 cents. Read
a bias of a few cents as "clean"; read tens of cents as a real offset worth correcting.

## Timestamp alignment

Predictions and labels are compared frame by frame on one shared grid, so a constant
timing error on either side silently biases every pitch number: a tracker whose stamps are
late reports the pitch of a slightly later moment, and the resulting cents error is
proportional to the local pitch slope. Both sides were measured and corrected.

**Trackers.** Each wrapper is probed with synthetic signals whose instantaneous f0 is known
analytically — a triangle chirp in log-frequency, whose slope alternates sign each leg, so
a constant *frequency* bias (which a tracker may legitimately have) separates by least
squares from a *timestamp* offset instead of contaminating it. A clock-rate error appears
as drift. A wrapper may shift its backend's stamps only when the measured error is
rate-invariant (the same across sample rates and sweep rates), probe-consistent (chirp and
step probes agree in sign and magnitude), and ideally traceable to the backend's frame
geometry. Errors failing those criteria are model behaviour and stay in the tracker's
score. Corrections are applied identically to every tracker, and each correction is
documented at the site that applies it.

**Datasets.** Annotations carry timing conventions too, and several were measurably off.
Each corpus's label offset was estimated by sweeping the label grid against multiple
independent trackers and taking the consensus — a shift that all trackers agree on is a
property of the labels, not of any one tracker. Where the consensus was systematic it is
corrected in the loader (`*_LABEL_OFFSET_SECONDS`), with the measured value at the site.
PTDB's shipped reference was ~22 ms early; MOCHA, CMUArctic, AVID and Vocadito each needed
single-digit-millisecond corrections. Corpora whose consensus was already within
measurement noise were left alone.

Residual timing error is held to about 2 ms, which is what makes `cents bias` readable: on
declining speech that bound corresponds to roughly a cent of apparent offset, so a bias of
a few cents is noise and tens of cents is a real defect.

## The six tracks

Each track is a complete situation of use, not a decomposition axis, so nothing is scored
twice. Diagnostics beneath each track explain it and are never scored.

| track | question | score |
|---|---|---|
| **Correctness** | how correct is the curve on clean real recordings? | pitch F1, equal weight per dataset |
| **Noise** | how well does it work under corruption? | absolute pitch F1 on the degraded conditions |
| **Signal types** | how much of the signal-class space is handled? | mean unconditional pitch recall over synthetic families |
| **Tracking** | is a moving pitch followed faithfully? | mean of steady-tone accuracy and vibrato depth |
| **Notes** | is musical note structure recoverable? | COnP |
| **Speed** | is it deployable? | 1/(1+RTF) |

**Overall** = the harmonic mean of the six, equal weights: a tracker is only as good as its
weakest situation of use. A zero in any track drives it to zero, and the report names which
track did it. A track with no cells makes Overall undefined rather than silently averaging
over fewer things.

Trackers a failed axis has pinned to Overall 0 are ordered among themselves by the harmonic
mean of the tracks that did *not* fail, so failing one axis ranks above failing everywhere.

### Why Noise is absolute, not a retention ratio

A ratio `F_degraded / F_clean` rises when clean performance *falls*, so a tracker can raise
it by sandbagging its own clean output. Measured: degrading clean from .95 to .80 at fixed
degraded performance moves the ratio from .789 to .938. The absolute score has no such
surface. Retention is still reported, as an unscored diagnostic, because "how much was
lost" is worth reading next to "how much is left".

### Why Tracking scores total error

Scoring de-biased jitter let a tracker emitting a *constant* an octave off score a perfect
1.000, beating an honest tracker with 8 cents of noise. The score uses the total error
`hypot(jitter, bias)`, so being consistently wrong is not rewarded for being consistent.

## Crashes

A cell that ran and died scores **0 in the track that owns it** — not skipped. Skipping
meant a tracker that segfaults on a corpus was graded only on the corpora it survived, so
its score was computed over a different dataset set than its neighbour's and the two were
printed side by side as if comparable.

The distinction is in the data: `metadata.crash_kind` records `exit -N` (killed by signal)
and `timeout > Ns`, both attributable to the algorithm, while a backend that was never
installed writes no cell at all and stays missing.

A crashed synthetic family is charged to the one track that owns its family type —
stationary to Signal types, trajectory to Tracking, controls to neither — so a single dead
cell is never counted twice.

## Excluded data

- **Score-grade pitch ground truth** (M4Singer): the labels are the notated note, not the
  performed f0. Voicing labels are sound, so the corpus is kept for voicing, but it is
  excluded from every pitch number and from theta\* selection. The exclusion lives in the
  scorer, so no consumer can disagree with it.
- **Sparse-voiced corpora** (OSFGlottis, AVID): mostly-unvoiced sessions whose precision
  denominators are dominated by silence false positives. Flagged in place, never dropped.

## Confidence intervals on the track scores

**Correctness** and **Noise** carry a 95% interval. Their resampling unit is the clip:
clusters are resampled *within* each dataset, that dataset's pitch F1 is recomputed, and the
replicate is the equal-per-dataset mean — the same weighting the score itself uses. A
dataset whose cell crashed contributes 0 to every draw, because it contributes 0 to the
score; without that the interval brackets a number the score never reports.

**Signal types, Tracking, Notes and Speed carry no interval.** Their units are ~24
families, 5 families, 3 datasets and a single measurement. A bootstrap over a handful of
units produces a wide, unstable number that reads like precision and is not. An absent
interval is the honest answer.

## Statistics

All confidence intervals are 95% paired cluster bootstraps over per-clip sufficient
statistics, clustered by speaker / singer / piece — never by clip, since correlated clips
would give false precision.

Every scored value is exactly recomputable from summed per-clip statistics, so the CI
machinery resamples and recomputes the identical formula it reports. The stored columns
are all counts or sums of cents, hence additive:

```
frames  n_exact  n_correct  n_wrong  n_octave
sum_cents  sum_signed_cents_correct  tp  fp  fn
```

A leaderboard compares many pairs without a multiple-comparisons correction, so read a
single statistical tie as descriptive rather than confirmatory.

## Named conventions

Everything else in the benchmark is counts. The complete list of chosen numbers:

- error-class tolerances: 10 / 50 / 200 cents
- steady-jitter normaliser `sigma0` = 10 cents
- speed mapping `1/(1+RTF)`
- equal weights across tracks in the Overall, and equal weight per dataset within a track
