"""
ilmt_streak_sf.py  --  single-frame streak detector for ILMT TDI frames
=======================================================================
Design follows Nir, Zackay & Ofek (2018, AJ 156, 229; arXiv:1806.04204):
  * remove point sources BEFORE the line search (their Sec. 4.2),
  * PSF-matched, inverse-variance-weighted Radon sums (their Eq. 3, 11),
  * Fast Radon Transform by shift-and-add (their Eq. 10) with partial
    transforms at every dyadic step for short streaks (their Sec. 4.3),
  * iterative removal of found streaks (their Sec. 4.4),
  * detection threshold taken from the empirical null (their Sec. 7).
Differences / additions (ours, NOT from the paper -- must be validated):
  * masked pixels get weight 0 instead of being cut out or noise-filled,
  * per-slope robust (median/MAD) null normalisation of the S/N map,
  * finite-segment refinement + per-candidate validation (artefact vetoes).

Coordinates: 0-based array indices, x = column, y = row (origin at the
lower-left when shown with imshow(origin='lower'), as DS9 does).
DS9/FITS pixel coordinates are 1-based:  x_ds9 = x + 1,  y_ds9 = y + 1.

STATUS: tested only on synthetic frames (see selftest()). Behaviour on
real ILMT frames is UNVERIFIED -- inspect every output in DS9.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field, asdict
import numpy as np
import sep
from scipy import ndimage as ndi
from scipy.optimize import curve_fit

sep.set_extract_pixstack(5_000_000)


# ----------------------------------------------------------------------------
@dataclass
class Config:
    bin_factor: int = 2            # 4096 -> 2048 (power of two needed by FRT)
    bkg_box: int = 128
    star_thresh_sigma: float = 3.0
    star_minarea: int = 6
    compact_axis_ratio: float = 2.0   # a/b below this => treated as star/galaxy
    compact_max_a: float = 60.0       # px; round objects of any plausible size are not streaks
    core_ratio: float = 8.0           # peak / median(segment): core+tail (bleed) vs flat (streak)
    bleed_dilate: int = 4
    mask_r_faint: float = 2.5         # mask ellipse scale (in sep a,b units)
    mask_r_mid: float = 4.0
    mask_r_bright: float = 7.0
    mid_peak_sigma: float = 20.0
    bright_peak_sigma: float = 200.0
    detrend_columns: bool = True      # TDI: residual column offsets (Nir Sec 4.5 idea)
    detrend_rows: bool = True
    min_len_native: int = 64          # shortest streak searched (native px)
    cov_min: float = 0.5              # min fraction of valid data along a line
    topk_per_step: int = 40
    z_candidate: float = 6.0          # PROVISIONAL: send to refinement/validation
    z_accept: float = 8.0             # PROVISIONAL: calibrate on your own null
    max_passes: int = 3
    max_cands_per_pass: int = 12
    # validation vetoes (reasons are always logged)
    min_sigma_ratio: float = 0.6      # transverse sigma / PSF sigma
    max_sigma_ratio: float = 6.0
    max_core_frac: float = 0.5        # fraction of segment within R of a bright core
    core_radius: float = 20.0
    bleed_attach_px: float = 25.0
    max_masked_frac: float = 0.6
    min_half_snr: float = 3.0
    max_border_frac: float = 0.8
    seed: int = 0


# ----------------------------------------------------------------------------
# 1. Preprocessing
# ----------------------------------------------------------------------------
def load_fits(path):
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        data = np.asarray(h[0].data, dtype=np.float32)   # big-endian -> native
        hdr = h[0].header.copy()
    return data, hdr


def preprocess(img, cfg: Config):
    """Background, RMS, star mask, PSF sigma, weights. No random numbers."""
    bad = ~np.isfinite(img)
    work = np.where(bad, 0, img).astype(np.float32)
    bkg = sep.Background(work, mask=bad, bw=cfg.bkg_box, bh=cfg.bkg_box, fw=3, fh=3)
    back = bkg.back().astype(np.float32)
    rms = np.maximum(bkg.rms().astype(np.float32), 1e-3)
    res = (work - back).astype(np.float32)
    res[bad] = 0

    obj, seg = sep.extract(res, cfg.star_thresh_sigma, err=rms, minarea=cfg.star_minarea,
                           mask=bad, deblend_nthresh=32, deblend_cont=0.005,
                           segmentation_map=True)
    a = np.maximum(obj['a'], 1e-3)
    b = np.maximum(obj['b'], 1e-3)
    compact = (a / b < cfg.compact_axis_ratio) & (a < cfg.compact_max_a)
    xp = np.clip(obj['xpeak'], 0, img.shape[1] - 1); yp = np.clip(obj['ypeak'], 0, img.shape[0] - 1)
    peak_sig = obj['peak'] / np.maximum(rms[yp, xp], 1e-3)

    mask = bad.copy()
    for sel, r in ((compact & (peak_sig < cfg.mid_peak_sigma), cfg.mask_r_faint),
                   (compact & (peak_sig >= cfg.mid_peak_sigma) & (peak_sig < cfg.bright_peak_sigma), cfg.mask_r_mid),
                   (compact & (peak_sig >= cfg.bright_peak_sigma), cfg.mask_r_bright)):
        if sel.any():
            sep.mask_ellipse(mask, obj['x'][sel], obj['y'][sel], a[sel], b[sel],
                             obj['theta'][sel], r=r)
    # Bright ELONGATED objects: a saturated star + bleed trail has one dominant core and a decaying
    # tail (peak/median >> 1); a genuine streak is roughly flat along its length (ratio ~1-3).
    tail = (~compact) & (peak_sig >= cfg.mid_peak_sigma)
    n_bleed = 0
    bleed_mask = np.zeros(img.shape, bool)
    if tail.any():
        ids = np.where(tail)[0] + 1                     # sep labels are index+1
        med = np.asarray(ndi.median(res, labels=seg, index=ids))
        ratio = obj['peak'][tail] / np.maximum(med, 1e-3)
        sel_ids = ids[ratio >= cfg.core_ratio]
        n_bleed = len(sel_ids)
        if n_bleed:
            m_b = np.isin(seg, sel_ids)
            bleed_mask = ndi.binary_dilation(m_b, iterations=cfg.bleed_dilate)
            mask |= bleed_mask
        tail_cores = np.zeros(len(obj), bool); tail_cores[sel_ids - 1] = True
    else:
        tail_cores = np.zeros(len(obj), bool)
    bright = (compact & (peak_sig >= cfg.bright_peak_sigma)) | tail_cores
    cores = np.c_[xp[bright], yp[bright]].astype(float)

    # PSF sigma from clean, round, moderately bright stars
    good = (compact & (a / b < 1.3) & (obj['flag'] == 0) &
            (peak_sig > 20) & (peak_sig < cfg.bright_peak_sigma))
    psf = float(np.median(np.sqrt(a[good] * b[good]))) if good.sum() >= 15 else 2.2
    psf = float(np.clip(psf, 1.2, 4.0))

    # Detrend residual column/row offsets using only unmasked pixels (upper-clipped).
    det = res.copy()
    mres = np.where(mask, np.nan, res)
    if cfg.detrend_columns:
        det -= _clipped_offset(mres, axis=0)[None, :]
    if cfg.detrend_rows:
        mres2 = np.where(mask, np.nan, det)
        det -= _clipped_offset(mres2, axis=1)[:, None]
    det[mask] = 0

    w = np.where(mask, 0.0, 1.0 / rms ** 2).astype(np.float32)
    bleed_dist = ndi.distance_transform_edt(~bleed_mask).astype(np.float32) if bleed_mask.any() else None
    return dict(bleed_dist=bleed_dist, img=work, back=back, rms=rms, res=res, det=det, mask=mask, w=w,
                psf=psf, cores=cores, n_src=len(obj), n_compact=int(compact.sum()), n_bleed=n_bleed)


def _clipped_offset(arr, axis, nsig=2.5, iters=3):
    """Robust per-column/row level: iterative clip, upper clip tighter than lower."""
    a = arr.copy()
    for _ in range(iters):
        med = np.nanmedian(a, axis=axis, keepdims=True)
        sd = 1.4826 * np.nanmedian(np.abs(a - med), axis=axis, keepdims=True)
        a = np.where((a - med) > nsig * sd, np.nan, a)
        a = np.where((a - med) < -4 * sd, np.nan, a)
    off = np.nanmedian(a, axis=axis)
    return np.nan_to_num(off, nan=0.0).astype(np.float32)


def matched_maps(det, w, psf):
    """Numerator / denominator images for the weighted PSF matched filter.
    num = G_sigma * (w*M)         (Nir Eq. 3 numerator, cross-correlation with PSF)
    den = G_sigma/sqrt2 * w       (variance of the above, ~ Nir Eq. 11)
    """
    num = ndi.gaussian_filter(det * w, psf, mode='constant').astype(np.float32)
    den = ndi.gaussian_filter(w, psf / np.sqrt(2.0), mode='constant').astype(np.float32)
    return num, den


def bin_sum(a, b):
    if b == 1:
        return a
    h, w = a.shape
    return a.reshape(h // b, b, w // b, b).sum(axis=(1, 3))


# ----------------------------------------------------------------------------
# 2. Fast Radon Transform with partial-step scanning (Nir et al. 2018, Sec. 3-4.3)
# ----------------------------------------------------------------------------
def frt_scan(num, den, m_min, cfg: Config, rng, orient):
    """num/den: (Ny, Nx) with Ny a power of two. Lines are summed along y.
    Returns list of candidate dicts in BINNED coordinates of the input arrays."""
    Ny, Nx = num.shape
    M = int(round(np.log2(Ny)))
    assert 1 << M == Ny, "active axis must be a power of two"
    pad = Ny
    X = Nx + 2 * pad
    Sn = np.zeros((Ny, 1, X), np.float32); Sn[:, 0, pad:pad + Nx] = num
    Sd = np.zeros((Ny, 1, X), np.float32); Sd[:, 0, pad:pad + Nx] = den
    dmed = float(np.median(den[den > 0])) if np.any(den > 0) else 0.0
    out = []
    for m in range(1, M + 1):
        off_p, off = (1 << (m - 1)) - 1, (1 << m) - 1
        ns, nb = (1 << (m + 1)) - 1, Ny >> m
        Nn = np.empty((nb, ns, X), np.float32)
        Nd = np.empty((nb, ns, X), np.float32)
        for j in range(-off, off + 1):
            f = int(np.trunc(j / 2.0))
            c = f if (j % 2 == 0) else f + (1 if j > 0 else -1)
            ip = f + off_p
            for S, N in ((Sn, Nn), (Sd, Nd)):
                tgt = N[:, j + off, :]
                tgt[:] = S[0::2, ip, :]
                sec = S[1::2, ip, :]
                if c == 0:
                    tgt += sec
                elif c > 0:
                    tgt[:, :X - c] += sec[:, c:]
                else:
                    tgt[:, -c:] += sec[:, :X + c]
        Sn, Sd = Nn, Nd
        if m < m_min:
            continue
        L = 1 << m
        valid = Sd >= cfg.cov_min * dmed * L
        z = np.full(Sn.shape, np.nan, np.float32)
        np.divide(Sn, np.sqrt(np.maximum(Sd, 1e-30)), out=z, where=valid)
        for g in np.array_split(np.arange(ns), min(ns, 48)):     # per-slope-group null
            v = z[:, g, :]
            v = v[np.isfinite(v)]
            if v.size < 1000:
                z[:, g, :] = np.nan
                continue
            if v.size > 200_000:
                v = rng.choice(v, 200_000, replace=False)
            med = np.median(v)
            sig = 1.4826 * np.median(np.abs(v - med)) + 1e-12
            z[:, g, :] = (z[:, g, :] - med) / sig
        flat = np.nan_to_num(z, nan=-1e9).ravel()
        K = min(cfg.topk_per_step, flat.size - 1)
        idx = np.argpartition(flat, -K)[-K:]
        for i in idx:
            zi = flat[i]
            if zi < cfg.z_candidate * 0.8:
                continue
            blk, sj, x0 = np.unravel_index(i, z.shape)
            j = sj - off
            ya, yb = blk * L, blk * L + L - 1
            xa = x0 - pad
            xb = xa + j * (L - 1) / L
            out.append(dict(z=float(zi), m=m, L=L, orient=orient, ya=ya, yb=yb, xa=xa, xb=xb))
        del z, flat, valid
    return out


def _to_native(c, b):
    """Convert candidate endpoints from binned/oriented to native (x, y)."""
    xa, ya, xb, yb = c['xa'], c['ya'], c['xb'], c['yb']
    if c['orient'] == 1:                     # transposed run: (active=x, passive=y)
        xa, ya, xb, yb = ya, xa, yb, xb
    f = lambda v: (v + 0.5) * b - 0.5
    return f(xa), f(ya), f(xb), f(yb)


def _collinear(s1, s2, perp_tol=10.0, ang_tol=2.0, gap_tol=150.0):
    (x1, y1, x2, y2), (u1, v1, u2, v2) = s1, s2
    d1 = np.array([x2 - x1, y2 - y1]); L1 = np.hypot(*d1)
    d2 = np.array([u2 - u1, v2 - v1]); L2 = np.hypot(*d2)
    if L1 < 1 or L2 < 1:
        return False
    d1 /= L1; d2 /= L2
    ang = np.degrees(np.arccos(np.clip(abs(d1 @ d2), 0, 1)))
    if ang > max(ang_tol, np.degrees(perp_tol / max(min(L1, L2), 1.0))):
        return False
    n = np.array([-d1[1], d1[0]])
    p = [abs((np.array(q) - np.array([x1, y1])) @ n) for q in ((u1, v1), (u2, v2))]
    if max(p) > perp_tol:
        return False
    t = [(np.array(q) - np.array([x1, y1])) @ d1 for q in ((u1, v1), (u2, v2))]
    lo, hi = min(t), max(t)
    return (lo <= L1 + gap_tol) and (hi >= -gap_tol)


def search_lines(num_n, den_n, cfg, rng, verbose=False):
    """Run FRT on both orientations; return NMS-merged native-pixel seeds."""
    b = cfg.bin_factor
    nb_, db_ = bin_sum(num_n, b), bin_sum(den_n, b)
    m_min = max(1, int(np.ceil(np.log2(max(cfg.min_len_native / b, 2)))))
    cands = []
    for orient, (n_, d_) in enumerate(((nb_, db_), (nb_.T.copy(), db_.T.copy()))):
        t0 = time.time()
        cands += frt_scan(n_, d_, m_min, cfg, rng, orient)
        if verbose:
            print(f"   FRT orient {orient}: {time.time()-t0:.1f}s, raw cands {len(cands)}")
    cands.sort(key=lambda c: -c['z'])
    kept = []
    for c in cands:
        seg = _to_native(c, b)
        if any(_collinear(seg, k['seed']) for k in kept):
            continue
        kept.append(dict(c, seed=seg))
        if len(kept) >= cfg.max_cands_per_pass:
            break
    return kept


# ----------------------------------------------------------------------------
# 3. Finite-segment refinement (native resolution)
# ----------------------------------------------------------------------------
def _line_samples(p, q, step=1.0):
    d = np.array(q) - np.array(p); L = np.hypot(*d)
    n = max(int(L / step) + 1, 2)
    t = np.linspace(0, 1, n)
    return p[0] + t * d[0], p[1] + t * d[1], d / L, L


def _clip_to_image(p, q, shape):
    """Extend the infinite line through p,q to the image rectangle."""
    h, w = shape
    p = np.array(p, float); d = np.array(q, float) - p
    d /= np.hypot(*d)
    ts = []
    for k, lim in ((0, w - 1), (1, h - 1)):
        if abs(d[k]) > 1e-9:
            ts += [(0 - p[k]) / d[k], (lim - p[k]) / d[k]]
    ts = sorted(ts)
    # the two middle intersections bound the inside segment
    cand = []
    for t in ts:
        pt = p + t * d
        if -1e-6 <= pt[0] <= w - 1 + 1e-6 and -1e-6 <= pt[1] <= h - 1 + 1e-6:
            cand.append(t)
    if len(cand) < 2:
        return None
    return p + min(cand) * d, p + max(cand) * d


def _best_segment(ns, ds, min_len):
    """Contiguous segment maximising sum(ns)/sqrt(sum(ds)). O(n^2) with cumsums."""
    n = len(ns)
    cn = np.concatenate([[0], np.cumsum(ns, dtype=np.float64)])
    cd = np.concatenate([[0], np.cumsum(ds, dtype=np.float64)])
    best = (-np.inf, 0, 0)
    for i in range(0, n - min_len):
        S = (cn[i + min_len:] - cn[i]) / np.sqrt(np.maximum(cd[i + min_len:] - cd[i], 1e-30))
        k = int(np.argmax(S))
        if S[k] > best[0]:
            best = (float(S[k]), i, i + min_len + k - 1)   # inclusive end index
    return best


def refine(seed, num, den, shape, cfg):
    """Optimise line offset/tilt on native arrays; trim to the best finite segment."""
    ext = _clip_to_image(seed[:2], seed[2:], shape)
    if ext is None:
        return None
    P, Q = ext
    d = (Q - P) / np.hypot(*(Q - P)); nrm = np.array([-d[1], d[0]])

    def score(dp, dq, with_profile=False):
        p, q = P + dp * nrm, Q + dq * nrm
        xs, ys, dd, L = _line_samples(p, q)
        coords = np.vstack([ys, xs])
        nn = ndi.map_coordinates(num, coords, order=1, mode='constant', cval=0.0)
        dn = ndi.map_coordinates(den, coords, order=1, mode='constant', cval=0.0)
        if with_profile:
            return nn, dn, xs, ys
        S, i0, i1 = _best_segment(nn, dn, int(min(cfg.min_len_native, len(nn) - 2)))
        return S

    best = (-np.inf, 0, 0)
    for step, span in ((2.0, 8.0), (0.5, 2.0)):
        c0 = (0.0, 0.0) if best[0] == -np.inf else (best[1], best[2])
        grid = np.arange(-span, span + 1e-9, step)
        for a in grid:
            for bq in grid:
                s = score(c0[0] + a, c0[1] + bq)
                if s > best[0]:
                    best = (s, c0[0] + a, c0[1] + bq)
    nn, dn, xs, ys = score(best[1], best[2], with_profile=True)
    S, i0, i1 = _best_segment(nn, dn, int(min(cfg.min_len_native, len(nn) - 2)))
    return dict(snr_theory=float(S), x1=float(xs[i0]), y1=float(ys[i0]),
                x2=float(xs[i1]), y2=float(ys[i1]),
                prof_n=nn, prof_d=dn, i0=i0, i1=i1, xs=xs, ys=ys)


# ----------------------------------------------------------------------------
# 4. Validation (artefact vetoes) -- every veto is logged, none is silent
# ----------------------------------------------------------------------------
def _gauss(u, a, u0, s, c):
    return a * np.exp(-0.5 * ((u - u0) / s) ** 2) + c


def validate(r, pre, cfg):
    x1, y1, x2, y2 = r['x1'], r['y1'], r['x2'], r['y2']
    xs, ys, d, L = _line_samples((x1, y1), (x2, y2))
    h, w = pre['res'].shape
    flags, info = [], {}
    # (a) masked fraction (how much of the track is actually supported by data)
    mk = ndi.map_coordinates(pre['mask'].astype(np.float32), [ys, xs], order=0, mode='nearest')
    info['masked_frac'] = float(mk.mean())
    if info['masked_frac'] > cfg.max_masked_frac:
        flags.append('mostly_masked')
    # (b) closeness to bright stellar cores (bleed trails / diffraction spikes)
    cores = pre['cores']
    if len(cores):
        from scipy.spatial import cKDTree
        dist, _ = cKDTree(cores).query(np.c_[xs, ys])
        info['core_frac'] = float((dist < cfg.core_radius).mean())
    else:
        info['core_frac'] = 0.0
    if info['core_frac'] > cfg.max_core_frac:
        flags.append('bleed_or_spike_suspect')
    # near-vertical track that starts at/near a masked core+tail object => charge-bleed tail that
    # fades below the mask isophote (TDI transfer axis = y, so bleed is vertical)
    ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
    info['bleed_end_dist'] = np.nan
    if pre['bleed_dist'] is not None:
        ex = np.clip(np.round([x1, x2]).astype(int), 0, w - 1); ey = np.clip(np.round([y1, y2]).astype(int), 0, h - 1)
        dd = float(pre['bleed_dist'][ey, ex].min())
        info['bleed_end_dist'] = dd
        if abs(ang - 90) < 8 and dd < cfg.bleed_attach_px:
            flags.append('bleed_tail_attached')
    # (c) transverse profile on the NON-detrended residual, unmasked samples only
    n = np.array([-d[1], d[0]])
    u = np.arange(-15, 15.01, 0.5)
    pos = np.arange(0, L, 2.0)
    P = np.stack([np.array([x1, y1]) + p * d for p in pos])
    prof_sum = np.zeros_like(u); prof_w = np.zeros_like(u)
    for k, uu in enumerate(u):
        q = P + uu * n
        ok = (q[:, 0] >= 0) & (q[:, 0] <= w - 1) & (q[:, 1] >= 0) & (q[:, 1] <= h - 1)
        if not ok.any():
            continue
        qq = q[ok]
        vals = ndi.map_coordinates(pre['res'], [qq[:, 1], qq[:, 0]], order=1)
        wt = ndi.map_coordinates(pre['w'], [qq[:, 1], qq[:, 0]], order=1)
        prof_sum[k] = (vals * wt).sum(); prof_w[k] = wt.sum()
    prof = np.where(prof_w > 0, prof_sum / np.maximum(prof_w, 1e-30), np.nan)
    info['u'], info['profile'] = u, prof
    try:
        good = np.isfinite(prof)
        p0 = [np.nanmax(prof) - np.nanmin(prof), 0.0, pre['psf'], np.nanmedian(prof)]
        popt, _ = curve_fit(_gauss, u[good], prof[good], p0=p0,
                            bounds=([0, -6, 0.3, -np.inf], [np.inf, 6, 15, np.inf]))
        info['amp'], info['sigma_perp'] = float(popt[0]), float(abs(popt[2]))
        info['sigma_ratio'] = info['sigma_perp'] / pre['psf']
        if info['sigma_ratio'] < cfg.min_sigma_ratio:
            flags.append('narrower_than_psf')
        if info['sigma_ratio'] > cfg.max_sigma_ratio:
            flags.append('implausibly_wide')
    except Exception:
        info['amp'] = info['sigma_perp'] = info['sigma_ratio'] = np.nan
        flags.append('profile_fit_failed')
    # (d) split-half consistency and signal concentration along the track
    nn, dn = r['prof_n'][r['i0']:r['i1'] + 1], r['prof_d'][r['i0']:r['i1'] + 1]
    mid = len(nn) // 2
    s1 = nn[:mid].sum() / np.sqrt(max(dn[:mid].sum(), 1e-30))
    s2 = nn[mid:].sum() / np.sqrt(max(dn[mid:].sum(), 1e-30))
    info['snr_half1'], info['snr_half2'] = float(s1), float(s2)
    if min(s1, s2) < cfg.min_half_snr:
        flags.append('one_sided_support')
    pos_n = np.clip(nn, 0, None)
    k = max(1, int(0.05 * len(pos_n)))
    info['top5pct_frac'] = float(np.sort(pos_n)[-k:].sum() / max(pos_n.sum(), 1e-30))
    # (e) border hugging
    nb = ((xs < 3) | (xs > w - 4) | (ys < 3) | (ys > h - 4)).mean()
    info['border_frac'] = float(nb)
    if nb > cfg.max_border_frac:
        flags.append('hugs_border')
    info['angle_deg'] = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)
    info['length_px'] = float(L)
    return flags, info


# ----------------------------------------------------------------------------
# 5. Orchestration (single frame)
# ----------------------------------------------------------------------------
def remove_segment(pre, num, den, seg, psf, extra=3.0):
    """Zero weights in a strip around the segment so it cannot be re-found."""
    x1, y1, x2, y2 = seg
    h, w = pre['res'].shape
    xs, ys, d, L = _line_samples((x1, y1), (x2, y2), step=0.5)
    m = np.zeros((h, w), bool)
    ix = np.clip(np.round(xs).astype(int), 0, w - 1); iy = np.clip(np.round(ys).astype(int), 0, h - 1)
    m[iy, ix] = True
    m = ndi.binary_dilation(m, structure=ndi.generate_binary_structure(2, 2),
                            iterations=int(np.ceil(3 * psf + extra)))
    pre['w'][m] = 0; pre['det'][m] = 0
    return m


def detect_frame(img, cfg: Config = None, verbose=True):
    cfg = cfg or Config()
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()
    pre = preprocess(img, cfg)
    if verbose:
        print(f"[pre] {pre['n_src']} sources ({pre['n_compact']} compact), PSF sigma={pre['psf']:.2f}px, "
              f"masked={pre['mask'].mean()*100:.1f}%, bright cores={len(pre['cores'])}, bleed-like objects masked={pre['n_bleed']}  ({time.time()-t0:.1f}s)")
    results = []
    for ipass in range(cfg.max_passes):
        num, den = matched_maps(pre['det'], pre['w'], pre['psf'])
        seeds = search_lines(num, den, cfg, rng, verbose=False)
        seeds = [s for s in seeds if s['z'] >= cfg.z_candidate]
        if verbose:
            print(f"[pass {ipass}] candidates >= {cfg.z_candidate}: {len(seeds)}  "
                  f"top z = {[round(s['z'],1) for s in seeds[:6]]}")
        if not seeds:
            break
        done_pass = []
        for s in seeds:
            if any(_collinear(d_, s['seed'], perp_tol=12.0, gap_tol=400.0) for d_ in done_pass):
                continue                      # same line as one already refined in this pass
            r = refine(s['seed'], num, den, img.shape, cfg)
            if r is None:
                continue
            done_pass.append((r['x1'], r['y1'], r['x2'], r['y2']))
            flags, info = validate(r, pre, cfg)
            accepted = (s['z'] >= cfg.z_accept) and not flags
            rec = dict(z_frt=s['z'], snr_theory=r['snr_theory'], x1=r['x1'], y1=r['y1'],
                       x2=r['x2'], y2=r['y2'], accepted=bool(accepted), flags=flags,
                       pass_id=ipass, **{k: v for k, v in info.items() if k not in ('u', 'profile')},
                       _profile=(info['u'], info['profile']))
            results.append(rec)
            remove_segment(pre, num, den, (r['x1'], r['y1'], r['x2'], r['y2']), pre['psf'])
    if verbose:
        print(f"[done] {time.time()-t0:.1f}s, {sum(r['accepted'] for r in results)} accepted / {len(results)} candidates")
    return results, pre


# ----------------------------------------------------------------------------
# 6. Annotated output
# ----------------------------------------------------------------------------
def annotate(img, results, pre, title="", save=None, show_mask=False):
    import matplotlib; 
    import matplotlib.pyplot as plt
    from astropy.visualization import ImageNormalize, AsinhStretch
    res = pre['res']
    lo, hi = np.nanpercentile(res, [1, 99.7])
    norm = ImageNormalize(vmin=lo, vmax=hi, stretch=AsinhStretch(a=0.2), clip=True)
    n = len(results)
    fig = plt.figure(figsize=(14, 7 + 3 * ((n + 2) // 3)))
    gs = fig.add_gridspec(1 + (n + 2) // 3, 3, height_ratios=[3] + [1.1] * ((n + 2) // 3))
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(res, origin='lower', cmap='gray', norm=norm)
    if show_mask:
        ax.imshow(np.ma.masked_where(~pre['mask'], pre['mask']), origin='lower', cmap='autumn', alpha=0.25)
    for i, r in enumerate(results):
        col = '#00ff66' if r['accepted'] else '#ff4040'
        ax.plot([r['x1'], r['x2']], [r['y1'], r['y2']], '-' if r['accepted'] else '--', color=col, lw=1.4)
        ax.text((r['x1'] + r['x2']) / 2, (r['y1'] + r['y2']) / 2, str(i), color=col, fontsize=11, weight='bold')
    ax.set_title(f"{title}   green=accepted  red=rejected/unconfirmed")
    for i, r in enumerate(results):
        a2 = fig.add_subplot(gs[1 + i // 3, i % 3])
        u, p = r['_profile']
        a2.plot(u, p, 'b.-', lw=0.8)
        a2.set_title(f"#{i} z={r['z_frt']:.1f} L={r['length_px']:.0f} sig_perp={r['sigma_perp']:.1f}\n"
                     f"{'ACCEPT' if r['accepted'] else ','.join(r['flags']) or 'z<accept'}", fontsize=8,
                     color='g' if r['accepted'] else 'r')
        a2.axvline(0, color='k', lw=0.4)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=110)
    return fig


# ----------------------------------------------------------------------------
# 7. Synthetic self-test (known truth). Run:  python ilmt_streak_sf.py
# ----------------------------------------------------------------------------
def make_synthetic(n=4096, seed=1, psf=2.2, noise=20.0, with_streaks=True):
    rng = np.random.default_rng(seed)
    img = rng.normal(0, noise, (n, n)).astype(np.float32)
    yy, xx = np.mgrid[0:n, 0:n]
    img += (3000 + 40 * np.sin(xx / 900.0) + 30 * yy / n).astype(np.float32)       # smooth sky
    img += rng.normal(0, 3, n).astype(np.float32)[None, :]                         # column offsets (TDI-like)
    truth = []

    def add_star(x, y, flux):
        r = int(6 * psf) + 1
        x0, y0 = int(round(x)), int(round(y))
        ys, xs = slice(max(y0 - r, 0), min(y0 + r + 1, n)), slice(max(x0 - r, 0), min(x0 + r + 1, n))
        gy, gx = np.mgrid[ys, xs]
        img[ys, xs] += (flux / (2 * np.pi * psf ** 2) * np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2 * psf ** 2))).astype(np.float32)

    for _ in range(2500):
        add_star(rng.uniform(5, n - 5), rng.uniform(5, n - 5), 10 ** rng.uniform(2.3, 4.6))
    for _ in range(25):                                   # bright stars with vertical bleed trails
        x, y = rng.uniform(50, n - 50), rng.uniform(300, n - 600)
        add_star(x, y, 3e6)
        L = int(rng.uniform(150, 450))
        img[int(y):int(y) + L, int(x) - 1:int(x) + 2] += 400.0 * np.exp(-np.arange(L) / (0.6 * L))[:, None]
    # row of stars in a line (alignment trap)
    for k in range(25):
        add_star(300 + 60 * k, 1500 + 0.35 * 60 * k, 8e4)

    def add_streak(x1, y1, x2, y2, amp):
        """Peak per-pixel amplitude `amp` on the centre line, Gaussian cross-profile sigma=psf."""
        d = np.array([x2 - x1, y2 - y1], float); L = np.hypot(*d); d /= L
        pad = int(6 * psf) + 2
        xa, xb = int(max(min(x1, x2) - pad, 0)), int(min(max(x1, x2) + pad, n - 1))
        ya, yb = int(max(min(y1, y2) - pad, 0)), int(min(max(y1, y2) + pad, n - 1))
        gy, gx = np.mgrid[ya:yb + 1, xa:xb + 1]
        t = (gx - x1) * d[0] + (gy - y1) * d[1]
        u = -(gx - x1) * d[1] + (gy - y1) * d[0]
        sig = amp * np.exp(-0.5 * (u / psf) ** 2) * ((t >= 0) & (t <= L))
        img[ya:yb + 1, xa:xb + 1] += sig.astype(np.float32)
        truth.append((x1, y1, x2, y2, amp))

    if with_streaks:
      add_streak(0.1*n, 0.15*n, 0.46*n, 0.37*n, 0.35 * noise)      # long, faint (peak 0.35 sigma/pixel)
      add_streak(0.73*n, 0.8*n, 0.78*n, 0.86*n, 0.9 * noise)     # short, fainter-than-obvious
      add_streak(0.61*n, 0.05*n, 0.62*n, 0.95*n, 1.2 * noise)      # near-vertical long
    return img, truth


def _seg_match(det, tr, tol=15):
    """Detection counts as recovering truth if both endpoints lie near the true line and overlap >50%."""
    x1, y1, x2, y2 = tr[:4]
    d = np.array([x2 - x1, y2 - y1]); L = np.hypot(*d); d /= L; n = np.array([-d[1], d[0]])
    P = np.array([[det['x1'], det['y1']], [det['x2'], det['y2']]]) - np.array([x1, y1])
    perp = np.abs(P @ n); t = np.sort(P @ d)
    ov = max(0, min(t[1], L) - max(t[0], 0)) / L
    return perp.max() < tol and ov > 0.5


def selftest(n=4096, save='selftest.png'):
    img, truth = make_synthetic(n)
    t0 = time.time()
    res, pre = detect_frame(img, Config())
    print(f"total {time.time()-t0:.0f}s")
    for i, r in enumerate(res):
        hit = [j for j, tr in enumerate(truth) if _seg_match(r, tr)]
        print(f" #{i} z={r['z_frt']:6.1f} snr_th={r['snr_theory']:6.1f} L={r['length_px']:6.0f} "
              f"ang={r['angle_deg']:5.1f} acc={r['accepted']!s:5} truth={hit} flags={r['flags']}")
    annotate(img, res, pre, "synthetic self-test", save=save)
    return res, truth


if __name__ == "__main__":
    selftest()