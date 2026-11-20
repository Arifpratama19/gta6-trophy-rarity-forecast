# GTA 6 Trophy Rarity Forecaster

Entry for the [make.fun bounty "Forecast how many GTA VI players earn each trophy in week one"](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) ($250).

[![gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Arifpratama19/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Arifpratama19/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

`main.py` forecasts, for each of Grand Theft Auto VI's 55 PlayStation 5 trophies, the percent of the players Exophase tracks who will have earned it at the target time — as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy. It is a plain Python 3.13 program using only the standard library, with no network access and no model calls, so it runs offline inside the grader's sandbox.

## How it works

A trophy's earn rate is a ratio — earners divided by players tracked — and the two halves move for different reasons, so the program forecasts them separately and divides at the end.

**Stale pages are not many observations.** Exophase refreshes its counts on its own schedule, hours apart, not continuously. The history is first collapsed into distinct snapshots: consecutive identical reads are one stale page, not a trend. On launch night the page sat unchanged at 638 players tracked for hours, then jumped to 3,868 in a single refresh, and every rate except the one at 100% fell — the newly tracked players had played less and held fewer trophies.

**When the history shows change, fit the change.** With two or more snapshots, each trophy's earner count and the players-tracked count are extrapolated in log time (`ln(1 + hours since release)`) by recency-weighted least squares, anchored on the last snapshot. A trophy's earner slope is shrunk toward the pooled slope across trophies, because a rare trophy with four earners cannot support its own slope. The forecast rate is forecast earners over forecast players tracked.

**When the history shows no change, expect a refresh.** With a single snapshot no trend can be fit, so the program assumes Exophase refreshes before the target: the tracked count jumps the way the launch-night refresh did (638 → 3,868) while earners grow only modestly — faster for early, common trophies, slower for late platinum and gold ones. How much of that dilution is applied depends on how stale the snapshot already is at the target against the ~4.5-hour cadence of the one refresh on record.

**The spread is honest about the horizon.** Quantiles are built on the logit scale around the forecast median, with a spread that grows with the horizon and with how noisy the fits are. A 90-minute practice horizon gets a spread of about 0.55 logits — far wider than the house baseline's 0.125, which is why the baseline's band covers none of the refreshed rates — and the 4.5-day final horizon gets a spread near 0.8.

**It never crashes and always writes a valid file.** Empty history, a single read, missing rates, a trophy nobody has earned yet, a missing target — each falls back sensibly, ultimately to persistence with a horizon-scaled spread. Output is exactly `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`: one row per trophy, five finite non-decreasing numbers from 0 to 100, well under the 65,536-byte cap.

## Running it

```sh
python3 main.py                       # every *.in in $MAKEFUN_INPUT_DIR (or /input)
python3 main.py /input/p00.in         # one case file
python3 main.py p00                   # one case by id
```

Outputs go to `$MAKEFUN_OUTPUT_DIR` (or `/output`) as `<id>.out`. Runtime is about 50 ms per case, against an 8,000 ms cap.

## Local checks

The kit ships no answer file, so the true practice answers are not available locally. What is documented about them — 3,868 players tracked at the answer, every rate but the 100% one lower, and the house baseline scoring about 1.51 with a typical miss of about 0.68 points — pins the answers at roughly 0.20× the last rates (earners grew about a third while the tracked count grew 6.06×). Against proxy answers spanning pure dilution (0.165×) to mild dilution (0.35×), this forecaster's mean pinball loss on the two practice cases is 0.12–0.21, against the house baseline's 0.95–1.69 under the same proxies — comfortably below the documented 1.51 baseline in every scenario.
