#!/usr/bin/env python3
"""Forecast Grand Theft Auto VI trophy earn rates on Exophase.

For every trophy of GTA VI on PS5, this program forecasts five quantiles
(5%, 25%, 50%, 75%, 95%) of its earn rate -- the percent of the players
Exophase tracks who have earned it -- at the case's target time.

Usage:
    python3 main.py [case-path | case-id | input-dir]

With no argument it reads every ``*.in`` file in ``$MAKEFUN_INPUT_DIR``
(or ``/input/``) and writes one ``<id>.out`` per case to
``$MAKEFUN_OUTPUT_DIR`` (or ``/output/``). With a case file path it
processes just that file; with a bare id it reads ``/input/<id>.in``.

The model, in one paragraph
---------------------------
A trophy's rate is a ratio, ``rate = earners / players_tracked * 100``,
and the two halves move for different reasons, so they are forecast
separately and divided at the end. Exophase does not refresh its counts
continuously: the page can sit unchanged for hours and then jump, and a
refresh mostly adds newly tracked players who have played less, so rates
usually fall at a refresh even while the earner counts keep rising.
The history is therefore first collapsed into distinct snapshots
(consecutive identical reads are one stale page, not many observations).
When the history holds more than one snapshot, each trophy's earner
count and the players-tracked count are extrapolated in log time --
``x = ln(1 + hours since release)`` -- by recency-weighted least
squares, anchored on the last snapshot, with each trophy's earner slope
shrunk towards the pooled slope (rare trophies have too few earners to
trust their own slope). When the history holds a single snapshot, no
trend can be fit, so a refresh is expected instead: the tracked count
is assumed to jump the way the launch-night refresh did, earners are
assumed to grow a little (faster for early, common trophies, slower
for late, rare ones), and the blend depends on how stale the snapshot
already is at the target. Quantiles are built on the logit scale around
the forecast median, with a spread that grows with the horizon and
with how noisy the fits are, so a long horizon honestly admits a wide
range. Anything unexpected falls back to persistence with a
horizon-scaled spread, and the program always writes a valid file.

Standard library only, no network, no model calls.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _norm_cdf(x):
    """Standard-normal CDF, via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p):
    """Inverse standard-normal CDF for 0 < p < 1, by bisection on the CDF.

    Bisection is slower than a rational approximation but obviously
    correct at every level, tails included, which matters more here
    than the microseconds it costs.
    """
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(rate):
    """logit of a percent, clamped to [0.01, 99.99] first."""
    p = clamp(rate, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(eta):
    """Percent from a logit value, kept inside (0, 100)."""
    if eta >= 0.0:
        p = 1.0 / (1.0 + math.exp(-eta)) if eta < 700.0 else 1.0
    else:
        e = math.exp(eta) if eta > -700.0 else 0.0
        p = e / (1.0 + e)
    return clamp(p * 100.0, 0.01, 99.99)


def parse_time(value):
    """Parse an ISO 8601 timestamp (with 'Z' or an offset) to aware UTC."""
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


# ---------------------------------------------------------------------------
# History handling
# ---------------------------------------------------------------------------

def collapse_snapshots(history):
    """Collapse consecutive identical reads into distinct snapshots.

    Returns a list of (time, players_tracked, rates) with one entry per
    distinct page state, in time order. The snapshot's time is the first
    read that showed it: a refresh happened somewhere between the previous
    read and this one, and the read times are the closest observable
    moments to it.
    """
    snaps = []
    prev_key = None
    for read in history or []:
        if not isinstance(read, dict):
            continue
        rates = read.get("rates")
        if not isinstance(rates, list):
            rates = []
        tracked = read.get("players_tracked")
        key = (tracked, tuple(rates))
        if key == prev_key:
            continue
        when = parse_time(read.get("at")) or parse_time(read.get("slot"))
        if when is None:
            continue
        try:
            tracked_num = float(tracked)
        except (TypeError, ValueError):
            tracked_num = 0.0
        snaps.append((when, tracked_num, list(rates)))
        prev_key = key
    return snaps


def last_rates(history, n_trophies):
    """The last non-null rate per trophy, scanning history backwards."""
    last = [None] * n_trophies
    for read in reversed(history or []):
        if not isinstance(read, dict):
            continue
        rates = read.get("rates")
        if not isinstance(rates, list):
            continue
        for i in range(min(n_trophies, len(rates))):
            if last[i] is None and isinstance(rates[i], (int, float)):
                last[i] = float(rates[i])
        if all(v is not None for v in last):
            break
    return last


def weighted_slope(xs, ys, ws):
    """Weighted least-squares slope (and intercept) of y on x.

    Returns (slope, intercept, weighted residual std). Degenerate input
    gives slope 0 and the weighted mean as intercept.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    wsum = sum(ws)
    if wsum <= 0.0:
        return 0.0, sum(ys) / n, 0.0
    xbar = sum(w * x for w, x in zip(ws, xs)) / wsum
    ybar = sum(w * y for w, y in zip(ws, ys)) / wsum
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        resid = math.sqrt(
            sum(w * (y - ybar) ** 2 for w, y in zip(ws, ys)) / wsum)
        return 0.0, ybar, resid
    sxy = sum(w * (x - xbar) * (y - ybar)
              for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    if n > 2:
        dof = max(1, n - 2)
        resid = math.sqrt(
            sum(w * (y - (intercept + slope * x)) ** 2
                for w, x, y in zip(ws, xs, ys)) / (wsum * dof / n))
    else:
        resid = 0.0
    return slope, intercept, resid


# ---------------------------------------------------------------------------
# The forecast
# ---------------------------------------------------------------------------

# Launch-night refresh, the one refresh the kit documents: the page sat
# unchanged at 638 players tracked, then refreshed to 3,868. Earners grew
# in the meantime, so the typical rate after a refresh is about a fifth
# of what it was, not a sixth. These are priors for a history that has
# never shown a refresh; a history with refreshes in it is fitted instead.
REFRESH_TRACKED_JUMP = 3868.0 / 638.0      # ~6.06x more players tracked
REFRESH_EARNER_GROWTH = 1.33              # earners grew ~a third meanwhile
REFRESH_CADENCE_HOURS = 4.5               # how stale a page gets before one
TIER_EARNER_FACTOR = {"platinum": 0.75, "gold": 0.90,
                      "silver": 1.00, "bronze": 1.05}


def quantiles_from_median_spread(median_rate, spread, levels):
    """Five (or len(levels)) quantiles on the logit scale around a median."""
    centre = logit_pct(median_rate)
    out = []
    prev = None
    for tau in levels:
        q = inv_logit_pct(centre + norm_ppf(tau) * spread)
        q = round(clamp(q, 0.01, 100.0), 4)
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def persistence_forecast(trophies, last, horizon_days, levels):
    """Fallback: rates stay put, spread widens like a random walk."""
    spread = max(0.5 * math.sqrt(max(horizon_days, 0.0)), 0.10)
    rows = []
    for i, trophy in enumerate(trophies):
        rate = last[i] if last[i] is not None else 1.0
        rows.append({"id": trophy["id"],
                     "q": quantiles_from_median_spread(rate, spread, levels)})
    return rows


def forecast_case(case):
    """Forecast one case dict, returning the list of trophy rows."""
    trophies = case.get("trophies") or []
    if not trophies:
        return []
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    levels = [float(t) for t in levels]
    history = case.get("history") or []

    at = parse_time(case.get("at"))
    target = parse_time(case.get("target"))
    if at is None or target is None:
        horizon_days = 0.0625
    else:
        horizon_days = max((target - at).total_seconds() / 86400.0, 1.0 / 1440.0)
    horizon_minutes = float(case.get("horizon_minutes") or horizon_days * 1440.0)
    horizon_days = max(horizon_minutes / 1440.0, horizon_days)

    n = len(trophies)
    last = last_rates(history, n)
    snaps = collapse_snapshots(history)

    # Clock for the trend fits: hours since the game's release day began.
    release = None
    game = case.get("game") or {}
    if isinstance(game, dict) and game.get("release_date"):
        release = parse_time(str(game["release_date"]) + "T00:00:00Z")
    if release is None and snaps:
        release = snaps[0][0]
    if release is None and at is not None:
        release = at

    def log_clock(when):
        if release is None or when is None:
            return 0.0
        hours = (when - release).total_seconds() / 3600.0
        return math.log1p(max(0.0, hours))

    # ---- Path 1: the history shows change -- fit earners and tracked. ----
    if len(snaps) >= 2 and target is not None:
        rows = _trend_forecast(trophies, last, snaps, log_clock, target,
                               horizon_days, levels)
        if rows is not None:
            return rows
        # The fit failed: fall through to persistence, not to the
        # single-snapshot refresh prior, which would misuse an old page.
        return persistence_forecast(trophies, last, horizon_days, levels)

    # ---- Path 2: one snapshot only -- expect a refresh before the target. ----
    if snaps and target is not None:
        rows = _refresh_forecast(trophies, last, snaps, target,
                                 horizon_days, levels)
        if rows is not None:
            return rows

    # ---- Path 3: nothing usable -- persistence. ----
    return persistence_forecast(trophies, last, horizon_days, levels)


def _trend_forecast(trophies, last, snaps, log_clock, target,
                   horizon_days, levels):
    """Extrapolate log earners and log players-tracked in log time."""
    try:
        times = [s[0] for s in snaps]
        xs = [log_clock(t) for t in times]
        x_last = xs[-1]
        x_target = log_clock(target)
        if x_target <= x_last:
            x_target = x_last + 1e-6
        # Recency weights: a snapshot 12 hours old counts half.
        t_last = times[-1]
        weights = [0.5 ** max(0.0, (t_last - t).total_seconds() / 43200.0)
                   for t in times]

        tracked = [s[1] for s in snaps]
        if any(v <= 0.0 for v in tracked):
            return None
        log_n = [math.log(v) for v in tracked]
        slope_n, _, resid_n = weighted_slope(xs, log_n, weights)
        slope_n = clamp(slope_n, 0.0, 2.5)
        n_last = tracked[-1]
        n_target = n_last * math.exp(slope_n * (x_target - x_last))
        if not (n_target > 0.0) or not math.isfinite(n_target):
            return None

        # Per-trophy earner points in log space (half an earner as a
        # pseudo-count, so a trophy nobody has yet still has a slope).
        per_trophy = []
        for i in range(len(trophies)):
            pts_x, pts_y, pts_w = [], [], []
            for k, snap in enumerate(snaps):
                rates = snap[2]
                if i < len(rates) and isinstance(rates[i], (int, float)):
                    earners = float(rates[i]) * snap[1] / 100.0
                    pts_x.append(xs[k])
                    pts_y.append(math.log(max(earners, 0.0) + 0.5))
                    pts_w.append(weights[k])
            per_trophy.append((pts_x, pts_y, pts_w))
        slopes = []
        resid_e_all = []
        for pts_x, pts_y, pts_w in per_trophy:
            if len(pts_x) < 2:
                continue
            b, _, r = weighted_slope(pts_x, pts_y, pts_w)
            slopes.append(b)
            if len(pts_x) >= 3:
                resid_e_all.append(r)
        pooled = median(slopes) if slopes else 0.0
        pooled = clamp(pooled, -0.5, 3.0)
        fit_noise = math.sqrt(
            (median(resid_e_all) if resid_e_all else 0.0) ** 2 + resid_n ** 2)
        fit_noise = min(fit_noise, 0.8)

        spread = math.sqrt(0.45 ** 2 + (0.30 * math.sqrt(horizon_days)) ** 2
                           + fit_noise ** 2)
        spread = clamp(spread, 0.30, 1.40)

        rows = []
        for i, trophy in enumerate(trophies):
            rate_last = last[i]
            if rate_last is not None and rate_last >= 99.99:
                rows.append({"id": trophy["id"],
                             "q": _hundred_row(levels)})
                continue
            earners_last = None
            if rate_last is not None:
                earners_last = rate_last * n_last / 100.0
            if earners_last is None:
                earners_last = 0.25
            pts_x, pts_y, pts_w = per_trophy[i]
            if len(pts_x) >= 2:
                b_i, _, _ = weighted_slope(pts_x, pts_y, pts_w)
                shrink = earners_last / (earners_last + 8.0)
                b_i = clamp(shrink * b_i + (1.0 - shrink) * pooled,
                            -0.5, 3.0)
            else:
                b_i = pooled
            base = earners_last if earners_last > 0.0 else 0.25
            earners_target = base * math.exp(b_i * (x_target - x_last))
            median_rate = clamp(100.0 * earners_target / n_target,
                                0.01, 99.99)
            rows.append({"id": trophy["id"],
                         "q": quantiles_from_median_spread(median_rate,
                                                           spread, levels)})
        return rows
    except Exception:
        return None


def _hundred_row(levels):
    """A trophy everyone holds: essentially 100%, with a token lower tail."""
    if len(levels) == 5:
        return [99.9, 99.98, 100.0, 100.0, 100.0]
    return quantiles_from_median_spread(99.99, 0.12, levels)


def _refresh_forecast(trophies, last, snaps, target, horizon_days, levels):
    """One snapshot only: expect Exophase to refresh before the target.

    The tracked count jumps (launch night: 638 -> 3,868) while earners
    grow only a little, so rates dilute. How much earners grow depends
    on the trophy: early, common trophies are picked up fast by new
    players, late platinum/gold ones barely at all. The blend follows
    how stale the snapshot already is at the target against the ~4.5
    hour cadence of the one refresh on record.
    """
    try:
        when0, n0, _ = snaps[-1]
        if n0 <= 0.0:
            return None
        age_hours = (target - when0).total_seconds() / 3600.0
        p_refresh = clamp(age_hours / REFRESH_CADENCE_HOURS, 0.0, 1.0)
        mixture = p_refresh * (1.0 - p_refresh) * (math.log(1.0 / 0.22) ** 2)
        spread = math.sqrt(0.55 ** 2 + (0.30 * math.sqrt(horizon_days)) ** 2
                           + mixture)
        spread = clamp(spread, 0.30, 1.40)

        rows = []
        for i, trophy in enumerate(trophies):
            rate = last[i] if last[i] is not None else 1.0
            if rate >= 99.99:
                rows.append({"id": trophy["id"],
                             "q": _hundred_row(levels)})
                continue
            tier = trophy.get("tier") if isinstance(trophy, dict) else None
            tier_f = TIER_EARNER_FACTOR.get(tier, 1.0)
            ease_f = clamp((max(rate, 0.05) / 0.78) ** 0.15, 0.80, 1.25)
            earner_growth = clamp(REFRESH_EARNER_GROWTH * tier_f * ease_f,
                                  1.05, 2.0)
            diluted = earner_growth / REFRESH_TRACKED_JUMP
            mult = (1.0 - p_refresh) + p_refresh * diluted
            median_rate = clamp(rate * mult, 0.01, 99.99)
            rows.append({"id": trophy["id"],
                         "q": quantiles_from_median_spread(median_rate,
                                                           spread, levels)})
        return rows
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def process_file(in_path, out_dir):
    """Read one case file, forecast it, write its output file."""
    with open(in_path, "r", encoding="utf-8") as fh:
        case = json.load(fh)
    case_id = case.get("id")
    if not isinstance(case_id, str) or not case_id:
        case_id = os.path.splitext(os.path.basename(in_path))[0]
    try:
        rows = forecast_case(case)
    except Exception:
        trophies = case.get("trophies") or []
        last = last_rates(case.get("history") or [], len(trophies))
        levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
        rows = persistence_forecast(trophies, last, 0.0625, levels)
    payload = {"trophies": rows}
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, case_id + ".out")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
        fh.write("\n")
    return out_path


def main(argv):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"

    targets = []
    if len(argv) > 1:
        arg = argv[1]
        if os.path.isdir(arg):
            targets = sorted(
                os.path.join(arg, name)
                for name in os.listdir(arg) if name.endswith(".in"))
        elif os.path.isfile(arg):
            targets = [arg]
        else:
            # A bare case id: read /input/<id>.in (or $MAKEFUN_INPUT_DIR).
            targets = [os.path.join(in_dir, arg + ".in")]
    else:
        if os.path.isdir(in_dir):
            targets = sorted(
                os.path.join(in_dir, name)
                for name in os.listdir(in_dir) if name.endswith(".in"))

    if not targets:
        sys.stderr.write("no case files found\n")
        return 1
    wrote = 0
    for path in targets:
        try:
            process_file(path, out_dir)
            wrote += 1
        except Exception as exc:  # one bad file must not kill the rest
            sys.stderr.write("failed %s: %s\n" % (path, exc))
    return 0 if wrote else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
