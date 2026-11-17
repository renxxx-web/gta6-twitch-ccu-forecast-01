# GTA VI Twitch CCU Forecaster — eve-anchored ridge blend

[![bounty](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/renxxx-web/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/renxxx-web/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for the make.fun bounty [gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01):
forecast, hour by hour, how many people watch Grand Theft Auto VI live on
Twitch in each hour of its release day after the first — five quantiles
(5/25/50/75/95) for hours 2–24, scored by mean pinball loss on the log scale.

## Model

`main.py` (Python 3.13, standard library only, no network) fits everything at
run time from the case's own comparables:

- Every comparable launch is rebuilt as a curve of log-ratios to its first
  hour, plus "eve" features from the 24 hours before release day — how large
  the pre-release audience already was relative to hour 1 (max, last-6h mean,
  last value). Rolling console launches build most of their audience the
  evening before the UTC day (New Zealand unlocks first), so the eve ratios
  are what separate a day that holds its first hour from one that ramps
  five-fold into the US evening.
- Comparables are weighted by similarity: archetype factors (simultaneous
  unlocks and early-access launches count less; a comparable whose first hour
  was pre-unlock noise counts much less), a log-size kernel and an
  eve-strength kernel.
- Per hour, two weighted ridge fits on the log-ratio are blended 1:2 in log
  space: a least-squares fit with weighted-residual quantiles, and a direct
  pinball (quantile) fit refined by iteratively reweighted least squares.
- For a sane rolling console-style launch, a fixed per-hour bias correction
  (measured on the four practice launches, halved toward zero) lifts the
  US-unlock hours (5–11) and US prime hours (17–24) and trims the small hours
  (2–4), where this archetype systematically sits off the pooled median.
- Degenerate inputs fall back to the house persistence shape.

On the four practice launches the entry scores a mean pinball loss of
**0.165**, against the house persistence baseline's 0.387 — a 57% improvement
(per case: Wolverine 0.263, RDR2 0.142, God of War Ragnarök 0.087,
Spider-Man 2 0.167). It runs in about half a second per case and is fully
deterministic.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

MIT licensed.
