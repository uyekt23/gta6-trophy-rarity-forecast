"""Forecast GTA VI PS5 trophy earn rates on Exophase at a future target time.

The program reads every case file in the input directory.  A case carries the
trophy list and every read the recorder made of the Exophase page before the
checkpoint.  For each trophy it writes five quantiles (5, 25, 50, 75 and 95
percent) of the earn rate, in percent, that the page is expected to show at
the case target.

Method, in short:

* Exophase refreshes its counts in steps, so consecutive reads often repeat
  one page state.  The history is first compressed to the distinct page
  states it actually contains; those states are the real observations.
* On the logit scale each trophy's rate is fitted against log elapsed time
  since release with a recency-weighted straight line.  Rates drift both
  ways: progression trophies climb as players advance, while easy early
  trophies are diluted when a wave of new tracked players joins.
* A trophy with a thin or noisy history borrows the average slope of its
  kind (story chapters, mission completions, tier challenges, activities),
  and story-chapter medians are kept in a soft decreasing order, because a
  later chapter cannot be earned by many more players than an earlier one.
* If the page has effectively never refreshed inside the observed window,
  the trend is unidentifiable and a refresh jump is the dominant risk: the
  median is shifted down (a bigger tracked-player pool dilutes almost every
  rate) and the spread is widened a lot.  This is the launch-night trap the
  practice cases are built around.
* The spread combines the fit residual, a random-walk term that grows with
  the square root of the horizon, and a refresh-jump allowance, so the
  5-95 band aims at honest coverage rather than false precision.

Only the Python standard library is used, the output is deterministic, and
any trophy whose modelling fails falls back to its last observed rate with
a wide spread instead of failing the case.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
RATE_FLOOR = 0.01
RATE_CEIL = 99.99
# Random-walk growth of a rate on the logit scale, per square-root day.
WALK_PER_ROOT_DAY = 0.62
# How strongly a thin history is pulled toward its kind's average slope.
SHRINK_STRENGTH = 3.0
# Recency weighting for the trend fit: a read this many hours old weighs half.
RECENCY_HALF_LIFE_H = 36.0


def clamp_rate(pct):
    return min(RATE_CEIL, max(RATE_FLOOR, pct))


def to_logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def from_logit(value):
    if value >= 40.0:
        return 100.0
    if value <= -40.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-value))


def round4(value):
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    result = whole / 10000.0
    return int(result) if float(result).is_integer() else result


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if p <= 0.0:
        return -3.5
    if p >= 1.0:
        return 3.5
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def parse_time(text):
    if not text:
        return None
    try:
        cleaned = text.strip().replace("Z", "+00:00")
        moment = datetime.fromisoformat(cleaned)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.timestamp()
    except (ValueError, AttributeError):
        return None


def release_epoch(case):
    game = case.get("game") or {}
    raw = game.get("release_date") or ""
    moment = parse_time(raw + "T00:00:00Z") if raw else None
    if moment is not None:
        return moment
    history = case.get("history") or []
    if history:
        first = parse_time(history[0].get("at") or history[0].get("slot"))
        if first is not None:
            return first
    return parse_time(case.get("at")) or 0.0


CHAPTER_RE = re.compile(r"chapter\s+(\d+)", re.IGNORECASE)


def trophy_kind(trophy):
    desc = (trophy.get("description") or "").strip()
    tier = trophy.get("tier") or ""
    if tier == "platinum":
        return "platinum"
    if CHAPTER_RE.search(desc) or desc.lower().startswith("complete the final mission"):
        return "story_chapter"
    if desc.lower().startswith("complete "):
        return "mission_complete"
    if tier in ("gold", "silver"):
        return "tier_challenge"
    return "activity"


def chapter_order(trophy):
    """Position inside the story sequence, or None for non-story trophies."""
    desc = (trophy.get("description") or "").strip()
    match = CHAPTER_RE.search(desc)
    if match:
        return int(match.group(1))
    if desc.lower().startswith("complete the final mission"):
        return 90
    return None


def distinct_states(history):
    """Compress the read history to the page states that actually changed.

    A state is kept when the tracked-player count or any trophy rate differs
    from the previous kept state.  Each kept entry is (epoch, players, rates,
    first_seen_epoch); first_seen records when that state first appeared,
    which is what makes a long-unchanged page recognisable as stale.
    """
    states = []
    for read in history:
        when = parse_time(read.get("at") or read.get("slot"))
        if when is None:
            continue
        players = read.get("players_tracked")
        rates = read.get("rates") or []
        if states and states[-1][1] == players and list(states[-1][2]) == list(rates):
            continue
        states.append((when, players, list(rates), when))
    return states


def weighted_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.  Returns (a, b, residual_sigma)."""
    total_w = sum(ws)
    if total_w <= 0.0:
        return (ys[-1] if ys else 0.0), 0.0, 0.35
    mean_x = sum(w * x for x, w in zip(xs, ws)) / total_w
    mean_y = sum(w * y for y, w in zip(ys, ws)) / total_w
    var_x = sum(w * (x - mean_x) ** 2 for x, w in zip(xs, ws)) / total_w
    if len(xs) < 3 or var_x < 1e-9:
        return mean_y, 0.0, 0.35
    cov = sum(w * (x - mean_x) * (y - mean_y) for x, y, w in zip(xs, ys, ws)) / total_w
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    resid = math.sqrt(
        sum(w * (y - (intercept + slope * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / total_w
    )
    return intercept, slope, max(resid, 0.05)


def fit_trophy(series, last_epoch):
    """Fit one trophy's logit trend from its (epoch, rate) observations."""
    xs, ys, ws = [], [], []
    for epoch, rate in series:
        if rate is None:
            continue
        hours = max((epoch - fit_trophy.release) / 3600.0, -0.9)
        xs.append(math.log1p(hours))
        ys.append(to_logit(rate))
        age_h = max((last_epoch - epoch) / 3600.0, 0.0)
        ws.append(0.5 ** (age_h / RECENCY_HALF_LIFE_H))
    if not ys:
        return None
    intercept, slope, resid = weighted_line(xs, ys, ws)
    flat = len(set(round(y, 9) for y in ys)) <= 1
    return {
        "intercept": intercept,
        "slope": slope,
        "resid": resid,
        "points": len(ys),
        "flat": flat,
        "last_logit": ys[-1],
        "last_x": xs[-1],
    }


fit_trophy.release = 0.0


def build_forecast(case):
    trophies = case.get("trophies") or []
    # Reads in parsed-timestamp order, with anything stamped after the
    # checkpoint removed before the model ever sees it.
    history = ordered_reads(case)
    levels = case.get("levels") or DEFAULT_LEVELS
    zscores = [norm_ppf(float(tau)) for tau in levels]

    checkpoint = parse_time(case.get("at"))
    target = parse_time(case.get("target"))
    if checkpoint is None and history:
        checkpoint = parse_time(history[-1].get("at") or history[-1].get("slot"))
    if checkpoint is None:
        checkpoint = 0.0
    horizon_min = case.get("horizon_minutes")
    if target is None and horizon_min is not None:
        target = checkpoint + float(horizon_min) * 60.0
    if target is None:
        target = checkpoint + 86400.0
    horizon_days = max((target - checkpoint) / 86400.0, 0.0)

    release = release_epoch(case)
    fit_trophy.release = release
    target_x = math.log1p(max((target - release) / 3600.0, -0.9))

    states = distinct_states(history)
    if not states and history:
        # History existed but nothing parsed: fall back to raw last read.
        last_rates = history[-1].get("rates") or []
        states = [(checkpoint, history[-1].get("players_tracked"), list(last_rates), checkpoint)]

    never_refreshed = len(states) <= 2
    last_epoch = states[-1][0] if states else checkpoint

    # Tracked-pool size: first and last valid counts, for the pinned-at-100
    # dilution allowance and the denominator-aware zero-trophy median.
    player_points = [(epoch, float(players)) for epoch, players, _r, _f in states
                     if isinstance(players, (int, float)) and not isinstance(players, bool)
                     and players > 0]
    players_last = player_points[-1][1] if player_points else None

    # Per-trophy fits.
    fits = []
    for idx in range(len(trophies)):
        series = [(epoch, valid_rate(rates[idx]) if idx < len(rates) else None)
                  for epoch, _p, rates, _f in states]
        try:
            fits.append(fit_trophy(series, last_epoch))
        except Exception:
            fits.append(None)

    # Kind-level and global slope pools, from trophies whose fit saw movement.
    def pooled(indices):
        num, den = 0.0, 0.0
        for i in indices:
            fit = fits[i]
            if fit and not fit["flat"] and fit["points"] >= 3:
                weight = min(fit["points"], 40)
                num += weight * fit["slope"]
                den += weight
        return (num / den) if den > 0 else None

    kinds = [trophy_kind(t) for t in trophies]
    global_slope = pooled(range(len(trophies)))
    kind_slope = {}
    for kind in set(kinds):
        kind_slope[kind] = pooled([i for i, k in enumerate(kinds) if k == kind])

    medians = []
    spreads = []
    overrides = {}
    for idx, trophy in enumerate(trophies):
        fit = fits[idx]
        last_rate = None
        if states:
            tail = states[-1][2]
            last_rate = valid_rate(tail[idx]) if idx < len(tail) else None
        if fit is None:
            # No observation at all: a faint positive guess with a wide tail.
            medians.append(to_logit(zero_median_pct(kinds[idx], players_last)))
            spreads.append(1.35)
            continue

        group = kind_slope.get(kinds[idx])
        anchor_slope = group if group is not None else global_slope
        if fit["flat"] or fit["points"] < 3:
            slope = 0.35 * anchor_slope if anchor_slope is not None else 0.0
            centre = fit["last_logit"] + slope * max(target_x - fit["last_x"], 0.0)
            resid = 0.45
        else:
            shrink = fit["points"] / (fit["points"] + SHRINK_STRENGTH)
            if anchor_slope is not None:
                slope = shrink * fit["slope"] + (1.0 - shrink) * anchor_slope
            else:
                slope = fit["slope"]
            centre = fit["intercept"] + slope * target_x
            resid = fit["resid"]

        if last_rate is not None and last_rate >= 99.99:
            # A printed 100% is a rounded value, not a certainty: it says
            # every tracked player had the trophy at the last refresh.  If
            # the tracked pool is growing, new joiners who have not earned
            # it yet dilute it, so the lower tail is allowed to fall: the
            # 5% quantile may drop to roughly 100 x N_now / N_projected,
            # with the projected pool growth capped, while the upper tail
            # stays at the ceiling.  With no observed pool growth the band
            # stays tight, as before.
            ratio = 1.0
            if len(player_points) >= 2:
                (t_first, p_first), (t_last, p_last) = player_points[0], player_points[-1]
                span_days = max((t_last - t_first) / 86400.0, 0.05)
                if p_last > p_first > 0:
                    daily = (p_last / p_first) ** (1.0 / span_days)
                    ratio = min(daily ** max(horizon_days, 0.0), 6.0)
            if ratio > 1.02:
                floor_pct = min(max(100.0 / ratio, 25.0), 99.5)
                mid_pct = min(max(100.0 / ratio ** 0.45, floor_pct + 0.5), 99.9)
                low, mid, high = to_logit(floor_pct), to_logit(mid_pct), to_logit(99.99)
                overrides[idx] = [
                    from_logit(low),
                    from_logit(low + 0.5 * (mid - low)),
                    from_logit(mid),
                    from_logit(mid + 0.55 * (high - mid)),
                    from_logit(high),
                ]
                medians.append(mid)
                spreads.append(0.55)
            else:
                medians.append(to_logit(99.99))
                spreads.append(0.55)
            continue
        if last_rate is not None and last_rate <= 0.0 and fit["flat"]:
            # Never earned in any read: a small positive median scaled to
            # the tracked pool (about one earner in it), wide upper tail.
            medians.append(to_logit(zero_median_pct(kinds[idx], players_last)))
            spreads.append(1.30)
            continue

        # Keep extrapolation sane: at most ~2.8 logit from the last reading.
        if fit["last_logit"] is not None:
            centre = min(max(centre, fit["last_logit"] - 2.8), fit["last_logit"] + 2.8)

        jump = 0.30
        if never_refreshed:
            # The page never moved in the window: a refresh before the target
            # is the main event, and a larger tracked pool dilutes rates.
            # Pinned and zero trophies were handled above.
            centre -= 1.15
            jump = 0.90
        walk = WALK_PER_ROOT_DAY * math.sqrt(horizon_days)
        sigma = math.sqrt(resid * resid + walk * walk + jump * jump)
        medians.append(centre)
        spreads.append(min(max(sigma, 0.38), 2.30))

    # Soft monotonicity along the story sequence: a later chapter's median
    # may sit at most a hair above the previous chapter's.
    sequence = []
    for idx, trophy in enumerate(trophies):
        order = chapter_order(trophy)
        if order is not None:
            sequence.append((order, idx))
    sequence.sort()
    previous = None
    for _order, idx in sequence:
        if previous is not None and medians[idx] > previous + 0.15:
            medians[idx] = previous + 0.15
        previous = medians[idx]

    rows = []
    for idx, trophy in enumerate(trophies):
        if idx in overrides and len(zscores) == 5:
            quantiles = sorted(round4(value) for value in overrides[idx])
        else:
            centre, sigma = medians[idx], spreads[idx]
            quantiles = []
            for z in zscores:
                quantiles.append(round4(from_logit(centre + z * sigma)))
            quantiles = sorted(quantiles)
        rows.append({"id": trophy.get("id", idx + 1), "q": quantiles})
    return {"trophies": rows}


def clean_number(value, default):
    """A finite float for value, or default when it is anything else."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def valid_rate(value):
    """Normalise one observed rate, or return None when it is not usable.

    Numbers and numeric strings are accepted (a string is converted);
    booleans, non-numeric values, non-finite numbers and anything outside
    0-100 are rejected, so one corrupt cell can never poison a trophy.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 100.0:
        return None
    return number


def ordered_reads(case):
    """The case's reads in parsed-UTC-timestamp order, future reads dropped.

    Reads are sorted by the timestamp they carry (stable, so equal stamps
    keep their file order; a read with no parseable stamp goes last), and
    any read stamped after the case's checkpoint is discarded — the model
    must never peek past the moment the forecast is made.
    """
    if not isinstance(case, dict):
        return []
    history = case.get("history")
    if not isinstance(history, list):
        return []
    checkpoint = parse_time(case.get("at"))
    indexed = []
    for pos, read in enumerate(history):
        if not isinstance(read, dict):
            continue
        when = parse_time(read.get("at") or read.get("slot"))
        if when is not None and checkpoint is not None and when > checkpoint:
            continue
        indexed.append((when, pos, read))
    indexed.sort(key=lambda item: (item[0] is None, item[0] or 0.0, item[1]))
    return [read for _when, _pos, read in indexed]


def zero_median_pct(kind, players_last):
    """A small positive median for a trophy never yet earned.

    The natural scale is one earner in the tracked pool: 100 / players,
    so the guess shrinks as the pool grows.  Platinum (which needs every
    other trophy) is scaled down further.  Floors and ceilings keep the
    value sane for tiny or enormous pools, and for an unknown pool size.
    """
    if players_last and players_last > 0:
        raw = 100.0 / float(players_last)
        if kind == "platinum":
            return min(max(raw * 0.4, 0.01), 0.75)
        return min(max(raw, 0.02), 1.50)
    return 0.06 if kind == "platinum" else 0.15


def sane_levels(case):
    """The case's quantile levels when they are usable, else the defaults."""
    levels = case.get("levels") if isinstance(case, dict) else None
    if isinstance(levels, list) and len(levels) == 5:
        cleaned = [clean_number(tau, -1.0) for tau in levels]
        if all(0.0 < tau < 1.0 for tau in cleaned) and cleaned == sorted(cleaned):
            return cleaned
    return list(DEFAULT_LEVELS)


def case_is_sound(case):
    """Defensive check of a case before the model trusts its contents.

    Sound means: a mapping with a non-empty trophy list whose entries carry
    distinct integer ids, and a non-empty history in which at least one read
    holds a rates list matching the trophy list in length.  Anything else —
    empty reads, a truncated rates row, duplicated ids, values of the wrong
    type — sends the case to the last-rate fallback instead of the model,
    so a damaged case degrades to a valid forecast rather than a crash.
    """
    if not isinstance(case, dict):
        return False
    trophies = case.get("trophies")
    if not isinstance(trophies, list) or not trophies:
        return False
    seen_ids = set()
    for trophy in trophies:
        if not isinstance(trophy, dict):
            return False
        tid = trophy.get("id")
        if not isinstance(tid, int) or isinstance(tid, bool) or tid in seen_ids:
            return False
        seen_ids.add(tid)
    history = case.get("history")
    if not isinstance(history, list) or not history:
        return False
    usable_reads = 0
    for read in history:
        if not isinstance(read, dict):
            return False
        rates = read.get("rates")
        if isinstance(rates, list) and len(rates) == len(trophies):
            usable_reads += 1
            for rate in rates:
                if rate is not None and valid_rate(rate) is None:
                    return False
    return usable_reads > 0


def fallback_forecast(case):
    """Last-rate persistence with a wide spread; used if modelling fails.

    Deliberately paranoid: every value it touches is sanitised first, so
    this path produces a well-formed forecast even for a case whose fields
    are missing, mistyped or half-corrupt.
    """
    if not isinstance(case, dict):
        case = {}
    trophies = case.get("trophies")
    if not isinstance(trophies, list):
        trophies = []
    history = case.get("history")
    if not isinstance(history, list):
        history = []
    zscores = [norm_ppf(tau) for tau in sane_levels(case)]
    horizon_days = max(clean_number(case.get("horizon_minutes"), 1440.0) / 1440.0, 0.01)
    sigma = max(0.7 * math.sqrt(horizon_days), 0.5)
    # Each trophy keeps its own last valid observation: the reads are walked
    # in timestamp order and a trophy's value is replaced only by a later
    # valid one, so a truncated or corrupt final read cannot erase the
    # earlier data of the trophies it fails to list.
    last_by_trophy = [None] * len(trophies)
    for read in ordered_reads(case):
        rates = read.get("rates")
        if not isinstance(rates, list):
            continue
        for idx in range(min(len(rates), len(trophies))):
            value = valid_rate(rates[idx])
            if value is not None:
                last_by_trophy[idx] = value
    rows = []
    for idx, trophy in enumerate(trophies):
        rate = last_by_trophy[idx] if last_by_trophy[idx] is not None else 0.5
        centre = to_logit(rate)
        quantiles = sorted(round4(from_logit(centre + z * sigma)) for z in zscores)
        tid = trophy.get("id") if isinstance(trophy, dict) else None
        rows.append({"id": tid if isinstance(tid, int) and not isinstance(tid, bool)
                      else idx + 1, "q": quantiles})
    return {"trophies": rows}


def enforce_shape(forecast, case):
    """Final pass over a forecast: nothing malformed reaches the disk.

    Rows come out sorted by trophy id, one per trophy the case lists, each
    with exactly five quantiles that are finite, rounded to 4 decimals,
    clamped to 0-100 and non-decreasing.  A row that is missing or broken
    is replaced by that trophy's fallback row.

    The endpoint clamp runs after rounding: no quantile is ever emitted as
    exactly 0 or exactly 100.  The scorer clamps rates into 0.01-99.99
    before taking logits, so an emitted endpoint would silently become a
    different forecast than the one written; every value is therefore held
    inside 0.01-99.99, which also keeps the 5% quantile at or above 0.01
    and the 95% quantile at or below 99.99.
    """
    fallback = fallback_forecast(case)
    filler = {row["id"]: row["q"] for row in fallback["trophies"]}
    offered = {}
    if isinstance(forecast, dict) and isinstance(forecast.get("trophies"), list):
        for row in forecast["trophies"]:
            if isinstance(row, dict) and isinstance(row.get("id"), int):
                offered.setdefault(row["id"], row.get("q"))
    rows = []
    for row in fallback["trophies"]:
        tid = row["id"]
        raw = offered.get(tid)
        quantiles = None
        if isinstance(raw, list) and len(raw) == 5:
            cleaned = [clean_number(value, None) for value in raw]
            if all(value is not None for value in cleaned):
                quantiles = []
                for value in cleaned:
                    # Round first, then clamp the rounded value into the
                    # scorer's own logit range: never exactly 0 or 100.
                    quantiles.append(min(max(round4(value), 0.01), 99.99))
                quantiles = sorted(quantiles)
        if quantiles is None:
            quantiles = sorted(min(max(round4(value), 0.01), 99.99)
                               for value in row["q"])
        rows.append({"id": tid, "q": quantiles})
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def write_atomic(path, text):
    """Write text to path atomically: a temp file in the same directory,
    flushed and closed, then renamed over the destination, so a reader (or
    a crash mid-write) never sees a half-written forecast."""
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_DIR.is_dir():
        return
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            # An unreadable case file names no trophies, so there is no
            # valid forecast to write for it; skip it and keep going.
            continue
        try:
            if case_is_sound(case):
                forecast = build_forecast(case)
            else:
                forecast = fallback_forecast(case)
        except Exception:
            forecast = fallback_forecast(case)
        forecast = enforce_shape(forecast, case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        write_atomic(OUTPUT_DIR / (path.stem + ".out"), text)


if __name__ == "__main__":
    main()
