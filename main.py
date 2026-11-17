"""gta6-twitch-ccu-forecast-01 — launch-day Twitch CCU forecaster.

Model (fit at run time from the case's own comparables, standard library only):

  A launch day is read as a curve of log-ratios to its first hour. For every
  past launch in the case's `comparables` we rebuild that curve, plus a small
  set of "eve" features describing the 24 hours before release day: how big
  the pre-release audience already was relative to hour 1 (its max, its mean
  over the last 6 hours, its last value). Rolling console launches build most
  of their audience before the UTC day even starts (New Zealand unlocks the
  evening before), so the eve ratios are what separate a day that merely
  holds its first hour from one that ramps five-fold into the US evening.

  Each comparable is weighted by similarity to the case: archetype factors
  (simultaneous-unlock and early-access launches count less, a comparable
  whose first hour was pre-unlock noise counts much less), a log-size kernel
  and an eve-strength kernel. For each forecast hour two weighted ridge fits
  on the log-ratio are blended: a least-squares fit whose quantiles come from
  the weighted residuals, and a direct pinball (quantile) fit refined by
  iteratively reweighted least squares. The blend is 1:2 in log space.

  Finally, when the case itself is a sane rolling console-style launch, a
  fixed per-hour bias correction is applied: measured on the four practice
  launches, this archetype's days sit above the pooled model's median in the
  US-unlock hours (5-11) and the US prime hours (17-24) and below it in the
  small hours (2-4). The constants are halved toward zero before use.

  If anything about the case is unusable (too few comparables, missing eve),
  the forecast falls back to the house persistence shape.
"""

import json
import math
import os
import sys
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
FEATS = ["r_evemax", "r_evemean6", "r_evelast", "logp1", "r_max_x_mean6"]
LAM = 1.0

# Per-hour archetype bias correction (log units), halved toward zero from the
# practice launches' residuals under this model.
CORR = {
    2: -0.033, 3: -0.022, 4: -0.046, 5: 0.136, 6: 0.211, 7: 0.219, 8: 0.144,
    9: 0.115, 10: 0.073, 11: 0.059, 12: -0.028, 13: 0.021, 14: 0.065,
    15: 0.057, 16: 0.085, 17: 0.172, 18: 0.260, 19: 0.170, 20: 0.134,
    21: 0.137, 22: 0.149, 23: 0.111, 24: 0.105,
}


def _ts(s):
    from datetime import datetime

    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _eve_feats(points):
    """points: [(t, viewers)] inside the 24 h before day_start, oldest first."""
    if not points:
        return None
    vals = [v for _, v in points]
    last6 = vals[-6:]
    return {
        "eve_max24": max(vals),
        "eve_mean6": sum(last6) / len(last6),
        "eve_mean24": sum(vals) / len(vals),
        "eve_last": vals[-1],
    }


def _ratio_feats(p1, eve):
    p1 = max(p1, 1.0)
    f = {"logp1": math.log(p1), "r_evemax": 0.0, "r_evemean6": 0.0, "r_evelast": 0.0}
    if eve:
        f["r_evemax"] = math.log(max(eve["eve_max24"], 1.0) / p1)
        f["r_evemean6"] = math.log(max(eve["eve_mean6"], 1.0) / p1)
        f["r_evelast"] = math.log(max(eve["eve_last"], 1.0) / p1)
    f["r_max_x_mean6"] = f["r_evemax"] * f["r_evemean6"]
    return f


def _day_curve(hourly, ds_ts):
    out = {}
    for p in hourly:
        v = p.get("viewers")
        if not v:
            continue
        h = int((_ts(p["from"]) - ds_ts) // 3600) + 1
        if 1 <= h <= 24:
            out[h] = float(v)
    return out


def _prehistory(hourly, ds_ts):
    pts = []
    for p in hourly:
        v = p.get("viewers")
        if not v:
            continue
        t = _ts(p["from"])
        if ds_ts - 24 * 3600 <= t < ds_ts:
            pts.append((t, float(v)))
    pts.sort()
    return pts


def _comp_features(c):
    ds_ts = _ts(c["day_start"])
    curve = _day_curve(c.get("hourly") or [], ds_ts)
    if len(curve) < 20 or 1 not in curve:
        return None
    plat = c.get("platforms") or ""
    unlock = (c.get("unlock") or "").lower()
    eve = _eve_feats(_prehistory(c.get("hourly") or [], ds_ts))
    return {
        "curve": curve,
        "p1": curve[1],
        "eve": eve,
        "rfeats": _ratio_feats(curve[1], eve),
        "console_only": "PC" not in plat,
        "simultaneous": "simultaneous" in unlock,
        "early_access": "early access" in unlock or "early-access" in unlock,
    }


def _weight(c, case):
    w = 1.0
    if c["simultaneous"]:
        w *= 0.15
    if c["early_access"]:
        w *= 0.3
    if c["eve"] and c["p1"] / max(c["eve"]["eve_mean6"], 1.0) < 0.45:
        w *= 0.1  # first hour was pre-unlock noise, not a real launch hour
    w *= math.exp(-abs(math.log(c["p1"]) - math.log(case["p1"])) / 1.5)
    if c["eve"] and case["eve"]:
        d = abs(
            math.log(c["eve"]["eve_max24"] / c["p1"])
            - math.log(case["eve"]["eve_max24"] / case["p1"])
        )
        w *= math.exp(-d / 0.7)
    return w


def _wquantile(pairs, tau):
    pairs = sorted(pairs)
    tot = sum(w for _, w in pairs)
    if tot <= 0:
        return pairs[len(pairs) // 2][0]
    target = tau * tot
    acc = 0.0
    for i, (v, w) in enumerate(pairs):
        acc += w
        if acc >= target:
            if i == 0:
                return v
            prev_v = pairs[i - 1][0]
            if w == 0:
                return v
            frac = (target - (acc - w)) / w
            return prev_v + frac * (v - prev_v)
    return pairs[-1][0]


def _ridge(X, y, keys, lam):
    n = len(keys)
    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for xi, yi in zip(X, y):
        xv = [xi[k] for k in keys]
        for i in range(n):
            b[i] += xv[i] * yi
            for j in range(n):
                A[i][j] += xv[i] * xv[j]
    for i in range(n):
        A[i][i] += lam * (1e-9 if keys[i] == "const" else 1.0)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        if abs(pv) < 1e-12:
            continue
        for j in range(col, n + 1):
            M[col][j] /= pv
        for r in range(n):
            if r != col and M[r][col] != 0:
                fac = M[r][col]
                for j in range(col, n + 1):
                    M[r][j] -= fac * M[col][j]
    return {keys[i]: M[i][n] for i in range(n)}


def _fit(rows, cf, lam):
    """rows: [(rfeats, target, weight)] -> (beta, [(resid, weight)])."""
    keys = ["const"] + FEATS
    X, y, ws = [], [], []
    for rf, t, w in rows:
        xi = {"const": 1.0}
        for k in FEATS:
            xi[k] = rf.get(k, cf.get(k, 0.0))
        X.append(xi)
        y.append(t)
        ws.append(w)
    Xw = [{k: v * math.sqrt(w) for k, v in xi.items()} for xi, w in zip(X, ws)]
    yw = [yi * math.sqrt(w) for yi, w in zip(y, ws)]
    beta = _ridge(Xw, yw, keys, lam)
    resid = [
        (yi - (beta["const"] + sum(beta[k] * xi[k] for k in FEATS)), w)
        for xi, yi, w in zip(X, y, ws)
    ]
    return beta, resid


def _predict(beta, cf):
    return beta["const"] + sum(beta[k] * cf.get(k, 0.0) for k in FEATS)


def _pinball_beta(rows, cf, tau):
    """Direct quantile fit: L2 start, then IRLS toward the pinball loss."""
    keys = ["const"] + FEATS
    beta, _ = _fit(rows, cf, LAM)
    for _ in range(12):
        iw = []
        for rf, t, w in rows:
            xi = {k: rf.get(k, cf.get(k, 0.0)) for k in FEATS}
            r = t - (beta["const"] + sum(beta[k] * xi[k] for k in FEATS))
            a = abs(tau - (1.0 if r < 0 else 0.0))
            iw.append(w * a / max(abs(r), 0.02))
        X, y = [], []
        for (rf, t, _), w in zip(rows, iw):
            xi = {"const": 1.0}
            for k in FEATS:
                xi[k] = rf.get(k, cf.get(k, 0.0))
            X.append({k: v * math.sqrt(w) for k, v in xi.items()})
            y.append(t * math.sqrt(w))
        beta = _ridge(X, y, keys, LAM * 0.05)
    return beta


def _persistence(p1):
    p = max(p1, 1.0)
    return {
        h: [p * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]
        for h in range(2, 25)
    }


def forecast(case):
    p1 = max(float(case["known"][0]["viewers"]), 1.0)
    ds_ts = _ts(case["day_start"])
    pts = []
    for p in case.get("history_hourly") or []:
        v = p.get("viewers")
        if not v:
            continue
        t = _ts(p["from"])
        if ds_ts - 24 * 3600 <= t < ds_ts:
            pts.append((t, float(v)))
    pts.sort()
    eve = _eve_feats(pts)
    case_f = {"p1": p1, "eve": eve, "rfeats": _ratio_feats(p1, eve)}

    comps = []
    for c in case.get("comparables") or []:
        if "before release" in (c.get("name") or ""):
            continue  # the case game's own pre-release category, not a launch
        cf = _comp_features(c)
        if cf is not None:
            comps.append(cf)
    if len(comps) < 6 or eve is None:
        return _persistence(p1)

    weighted = []
    for c in comps:
        w = _weight(c, case_f)
        if w > 1e-9:
            weighted.append((c, w))
    if len(weighted) < 6:
        return _persistence(p1)

    cf = case_f["rfeats"]
    out = {}
    for h in range(2, 25):
        rows = [
            (c["rfeats"], math.log(c["curve"][h] / c["p1"]), w)
            for c, w in weighted
            if h in c["curve"]
        ]
        if not rows:
            out[h] = [p1] * 5
            continue
        beta, resid = _fit(rows, cf, LAM)
        pred = _predict(beta, cf)
        q_l2 = [pred + _wquantile(resid, tau) for tau in LEVELS]
        q_pb = []
        for tau in LEVELS:
            btau = _pinball_beta(rows, cf, tau)
            q_pb.append(_predict(btau, cf))
        qs = [
            p1 * math.exp((a + 2.0 * b) / 3.0) for a, b in zip(q_l2, q_pb)
        ]
        for i in range(1, 5):
            qs[i] = max(qs[i], qs[i - 1])
        out[h] = qs

    # Archetype bias correction for sane rolling console-style launches.
    notes = (case.get("notes") or "").lower().replace("no simultaneous", "")
    sane = p1 / max(eve["eve_mean6"], 1.0) >= 0.45
    rolling = "rolling" in notes or "midnight local" in notes
    if sane and rolling and "simultaneous" not in notes:
        for h in range(2, 25):
            f = math.exp(CORR[h])
            out[h] = [q * f for q in out[h]]
    return out


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        rows = forecast(case)
        hours = []
        for h in range(2, 25):
            qs = []
            for q in rows[h]:
                q = min(max(q, 0.01), 1e9)
                qs.append(round(q, 2))
            for i in range(1, 5):
                qs[i] = max(qs[i], qs[i - 1])
            hours.append({"hour": h, "q": qs})
        out_path = out_dir / (path.stem + ".out")
        out_path.write_text(json.dumps({"hours": hours}), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
