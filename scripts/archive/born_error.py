#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
born_error.py -- measure the ISS truncation error of a multislice
ptychography reconstruction.

WHAT THIS COMPUTES
------------------
For a given slice stack {O_j}, probe P, geometry (lam, pixel size, slice
positions z_j) and a set of scan positions, this evaluates BOTH forward models
at every position and reports Eq. (epsilon) of the draft,

    eps_M = || psi_MS - psi^(<=M) || / || psi^(<=M) ||                     (1)

where psi_MS is the exact multislice exit wave, Eq. (cascade), and psi^(<=M) is
the scattering-order expansion, Eq. (hierarchy), truncated at order M. M = 1 is
the model of Eq. (model); M = N reproduces multislice identically.

Three families of numbers come out:

  1. eps_M for M = 0, 1, ..., max_order.  The individual orders psi^(m) are
     obtained *exactly* and without combinatorics by the roots-of-unity trick.

  2. eps_1 for the BLOCKED truncation of Sec. "Blocked truncation": slices
     grouped into M contiguous blocks.

  3. The dose criterion, Eq. (noisecrit).  A relative field error eps gives a
     relative intensity error 2 eps, invisible against a shot-noise floor.

It also computes the "Square of Errors" to isolate weak-phase vs truncation:
  - Exact Multislice (Baseline)
  - ISS with exact g = O - 1 (Truncation error only)
  - Multislice with tangent g = 1 + is (Weak-phase error only)
  - ISS with tangent g = is (Combined error)

USAGE
-----
(A) Exact, from inside a live ptypy session:
        import ptypy, born_error
        P = ptypy.core.Ptycho(pars, level=4)
        res = born_error.epsilon_from_ptycho(P, plot=True, out='eps')

(B) From files, no ptypy import needed. Defaults to NanoMAX geometry:
        python born_error.py --plot --out eps_434
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

try:
    from ptypy import utils as pu
    _PTYPY_ERROR = None
except Exception as _e:  # pragma: no cover
    pu = None
    _PTYPY_ERROR = _e


# ===========================================================================
#  Geometry and propagation
# ===========================================================================

NANOMAX = dict(
    basedir='/home/dnz75396/ptypy/data/',
    sample='0002_multislice',
    scannr=434,
    detector='eiger4m',
    det_psize=75e-6,
    cropping=512,
    binning=1,
    distance_m=4.18,
    slice_thickness=1580e-6,
    number_of_slices=2,
    probe_modes=2,
    niter=200,
    batch_size=64,
)

def default_paths(basedir=NANOMAX['basedir'], sample=NANOMAX['sample'],
                  scannr=NANOMAX['scannr'], cropping=NANOMAX['cropping'],
                  distance_m=NANOMAX['distance_m'],
                  slice_thickness=NANOMAX['slice_thickness'],
                  number_of_slices=NANOMAX['number_of_slices'],
                  niter=NANOMAX['niter']):
    out_dir = (f'{str(basedir).rstrip("/")}/process/{sample}/scan_{scannr:0>6}/'
               f'ptycho_ptypy_crop_{cropping}_detdist_{distance_m}_'
               f'slice_{slice_thickness}/')
    rec = out_dir + 'rec/'
    return dict(
        out_dir=out_dir,
        rec=rec,
        ptyd=out_dir + f'data/data_scan_{scannr:0>6}.ptyd',
        fslices=rec + (f'rec_crop-{cropping}_slices-{number_of_slices}'
                       f'_iter-{niter:0>4}.h5'),
        ptyr_glob=rec + f'rec_scan_{scannr:0>6}_*.ptyr',
        dumps_glob=out_dir + f'dumps/dump_scan_{scannr:0>6}_*.ptyr',
    )


def _iter_of(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    tail = stem.rsplit('_', 1)[-1]
    return int(tail) if tail.isdigit() else -1


def _engine_of(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.rsplit('_', 2)
    return parts[-2] if len(parts) >= 2 else ''


def pick_ptyr(pattern, engine=None, prefer='iterations', verbose=True):
    import glob
    hits = glob.glob(pattern)
    if not hits:
        return None
    if engine:
        sel = [h for h in hits if engine.lower() in _engine_of(h).lower()]
        if sel:
            hits = sel
        else:
            print("[warn] no .ptyr matches engine '%s'; using all %d"
                  % (engine, len(hits)))
    key = _iter_of if prefer == 'iterations' else os.path.getmtime
    hits.sort(key=key)
    if verbose and len(hits) > 1:
        print("[paths] %d candidate .ptyr files:" % len(hits))
        for h in hits:
            print("          %-28s iter %5d   %s"
                  % (_engine_of(h), _iter_of(h), os.path.basename(h)))
    return hits[-1]


def newest(pattern):
    import glob
    hits = sorted(glob.glob(pattern), key=os.path.getmtime)
    return hits[-1] if hits else None


def wavelength_from_geometry(resolution, shape, det_psize, distance):
    res = np.asarray(resolution, dtype=float).reshape(-1)[0]
    sh = np.asarray(shape).reshape(-1)[0]
    return float(res * sh * float(det_psize) / float(distance))


def wavelength_from_photon_kev(e_kev: float) -> float:
    return 1.23984193e-9 / float(e_kev)


def wavelength_from_electron_kv(u_kv: float) -> float:
    h, m0, e, c = 6.62607015e-34, 9.1093837015e-31, 1.602176634e-19, 299792458.0
    v = float(u_kv) * 1e3
    return h / np.sqrt(2 * m0 * e * v * (1.0 + e * v / (2 * m0 * c ** 2)))


class AngularSpectrum:
    def __init__(self, shape, resolution, lam, device, cdtype, rdtype):
        ny, nx = int(shape[0]), int(shape[1])
        res = np.asarray(resolution, dtype=float).reshape(-1)
        if res.size == 1:
            res = np.repeat(res, 2)
        py, px = float(res[0]), float(res[1])
        sh = np.array([ny, nx])
        if pu is not None:
            V, W = pu.grids(tuple(int(s) for s in sh), lam / sh / res[:2], 'fft')
            a2 = V ** 2 + W ** 2
        else:
            fy = lam * np.fft.fftfreq(ny, d=py)
            fx = lam * np.fft.fftfreq(nx, d=px)
            a2 = fy[:, None] ** 2 + fx[None, :] ** 2
        self.lam = float(lam)
        self.shape = (ny, nx)
        self.psize = (py, px)
        self._a2 = a2
        self._kz = np.sqrt(np.maximum(1.0 - a2, 0.0)) - 1.0
        self._evan = a2 > 1.0
        self.device, self.cdtype, self.rdtype = device, cdtype, rdtype
        self._cache = {}
        self.na_max = float(np.sqrt(np.clip(a2.max(), 0.0, 1.0)))

    def H(self, dz):
        key = round(float(dz), 15)
        H = self._cache.get(key)
        if H is None:
            Hn = np.exp(2j * np.pi * (key / self.lam) * self._kz)
            Hn[self._evan] = 0.0
            H = torch.as_tensor(Hn, dtype=self.cdtype, device=self.device)
            self._cache[key] = H
        return H

    def prop(self, field, dz):
        if abs(float(dz)) == 0.0:
            return field
        return torch.fft.ifft2(self.H(dz) * torch.fft.fft2(field))

    def __call__(self, field, dz):
        return self.prop(field, dz)


def _op_call(op, arr):
    last = None
    for dt in (np.complex64, np.complex128):
        try:
            return np.asarray(op(arr.astype(dt)), dtype=np.complex128)
        except Exception as e:
            last = e
    raise RuntimeError("could not call propagator: %s" % last)


def _apply_numpy(kind, d, w):
    if kind == 'lsi':
        return np.fft.ifft2(np.fft.fft2(w) * d['H'])
    if kind == 'ilsi':
        return np.fft.fft2(np.fft.ifft2(w) * d['H'])
    if kind == 'chirp':
        return d['A'] * np.fft.fft2(d['R'] * w)
    return d['A'] * np.fft.ifft2(d['R'] * w)


def _extract_operator(op, shape, name, rtol=1e-5):
    ny, nx = shape
    d00 = np.zeros(shape, dtype=np.complex128)
    d00[0, 0] = 1.0
    out00 = _op_call(op, d00)
    out1 = _op_call(op, np.ones(shape, dtype=np.complex128))

    cands = [('lsi', {'H': np.fft.fft2(out00)}),
             ('ilsi', {'H': float(ny * nx) * np.fft.ifft2(out00)})]
    if np.abs(out00).min() > 0:
        cands += [('chirp', {'A': out00, 'R': np.fft.ifft2(out1 / out00)}),
                  ('ichirp', {'A': out00, 'R': np.fft.fft2(out1 / out00)})]

    rng = np.random.default_rng(0)
    w = (rng.standard_normal(shape)
         + 1j * rng.standard_normal(shape)).astype(np.complex128)
    ref = _op_call(op, w)
    scale = max(np.abs(ref).max(), 1e-30)

    errs = {k: np.abs(_apply_numpy(k, d, w) - ref).max() / scale
            for k, d in cands}
    kind = min(errs, key=errs.get)
    if errs[kind] > rtol:
        raise RuntimeError("%s: propagator extraction failed, errors %s"
                           % (name, {k: '%.2e' % v for k, v in errs.items()}))
    spec = {'kind': kind}
    spec.update(dict(cands)[kind])
    print("[prop] %s extracted as '%s' (error %.2e)" % (name, kind, errs[kind]))
    return spec


class FarField:
    def __init__(self, spec=None, device=None, cdtype=None):
        self.spec = None
        if spec is not None:
            self.spec = {'kind': spec['kind']}
            for k in ('H', 'A', 'R'):
                if k in spec:
                    self.spec[k] = torch.as_tensor(spec[k], dtype=cdtype,
                                                   device=device)

    @classmethod
    def from_ptypy(cls, fw, shape, device, cdtype, name='farfield fw'):
        return cls(_extract_operator(fw, tuple(int(s) for s in shape), name),
                   device, cdtype)

    def __call__(self, w):
        s = self.spec
        if s is None:
            return torch.fft.fft2(w, norm='ortho')
        k = s['kind']
        if k == 'lsi':
            return torch.fft.ifft2(torch.fft.fft2(w) * s['H'])
        if k == 'ilsi':
            return torch.fft.fft2(torch.fft.ifft2(w) * s['H'])
        if k == 'chirp':
            return s['A'] * torch.fft.fft2(s['R'] * w)
        return s['A'] * torch.fft.ifft2(s['R'] * w)


def detector(psi):
    return torch.fft.fft2(psi, norm='ortho')


# ===========================================================================
#  Forward models
# ===========================================================================

def multislice_exit(O, P, prop, z):
    """Eq. (cascade): psi = O_{N-1} P_d ... O_1 P_d O_0 P."""
    psi = O[0][:, None] * P[None]
    for j in range(1, len(z)):
        psi = prop(psi, z[j] - z[j - 1]) * O[j][:, None]
    return psi


def illumination(P, prop, z):
    """Eq. (phi): phi_j = P_{z_j} P. Object free and position free."""
    return torch.stack([prop(P, zj) for zj in z])          # (N, M, ny, nx)


def born_exit(g, P, phi, prop, z):
    """Eq. (born-real), in the exit plane:
        psi = P_{z_{N-1}} P + sum_j P_{z_{N-1}-z_j} [ g_j phi_j ].
    """
    zN = z[-1]
    psi = prop(P, zN)[None].expand(g.shape[1], *P.shape).clone()
    for j in range(len(z)):
        psi = psi + prop(g[j][:, None] * phi[j][None], zN - z[j])
    return psi


def blocked_exit(g, P, prop, z, n_blocks):
    """Blocked truncation of Sec. (blocked)"""
    N = len(z)
    n_blocks = int(max(1, min(n_blocks, N)))
    blocks = np.array_split(np.arange(N), n_blocks)

    psi = P[None]
    z_in = z[0]
    for b, idx in enumerate(blocks):
        z_out = z[blocks[b + 1][0]] if b + 1 < len(blocks) else z[-1]
        out = prop(psi, z_out - z_in)
        for j in idx:
            inc = prop(psi, z[j] - z_in)
            out = out + prop(g[j][:, None] * inc, z_out - z[j])
        psi, z_in = out, z_out
    return psi


def order_fields(O, P, prop, z, max_order):
    """Exact scattering-order decomposition, Eq. (hierarchy)."""
    N = len(z)
    g = O - 1.0
    K = N + 1
    max_order = int(min(max_order, N))

    orders = None
    psi_ms = None
    for k in range(K):
        t = complex(np.exp(2j * np.pi * k / K))
        tt = torch.as_tensor(t, dtype=O.dtype, device=O.device)
        psik = multislice_exit(1.0 + tt * g, P, prop, z)
        if k == 0:
            psi_ms = psik.clone()
            orders = torch.zeros((max_order + 1,) + tuple(psik.shape),
                                 dtype=psik.dtype, device=psik.device)
        for m in range(max_order + 1):
            w = torch.as_tensor(complex(np.conj(t) ** m) / K,
                                dtype=psik.dtype, device=psik.device)
            orders[m] += w * psik
        del psik
    return psi_ms, orders


# ===========================================================================
#  The scan
# ===========================================================================

@dataclass
class ScanResult:
    eps_order: dict = field(default_factory=dict)
    eps_block: dict = field(default_factory=dict)
    eps1_per_pos: np.ndarray = None
    eps1lin_per_pos: np.ndarray = None
    phi_per_pos: np.ndarray = None
    meta: dict = field(default_factory=dict)
    example: dict = field(default_factory=dict)

    def as_dict(self):
        d = dict(self.meta)
        d['eps_order'] = {str(k): float(v) for k, v in self.eps_order.items()}
        d['eps_block'] = {str(k): float(v) for k, v in self.eps_block.items()}
        if self.eps1_per_pos is not None and len(self.eps1_per_pos):
            e = self.eps1_per_pos
            d['eps1_per_position'] = dict(
                median=float(np.median(e)), p05=float(np.percentile(e, 5)),
                p95=float(np.percentile(e, 95)), max=float(e.max()))
        if self.phi_per_pos is not None and len(self.phi_per_pos):
            p = self.phi_per_pos
            d['phi_tot'] = dict(rms=float(np.sqrt((p ** 2).mean())),
                                median=float(np.median(p)),
                                p95=float(np.percentile(p, 95)))
        return d


class BornErrorScan:
    def __init__(self, slices, probe, dlow, psize, lam, z,
                 device='cuda', precision='single', gauge='none',
                 measured=None, mask=None, farfield=None):
        if torch is None:
            raise ImportError("born_error requires PyTorch")

        dev = str(device)
        if dev.startswith('cuda') and not torch.cuda.is_available():
            print("[warn] CUDA unavailable, falling back to CPU")
            dev = 'cpu'
        self.dev = torch.device(dev)
        dbl = str(precision).lower().startswith('d')
        self.cdt = torch.complex128 if dbl else torch.complex64
        self.rdt = torch.float64 if dbl else torch.float32

        slices = [np.asarray(s) for s in slices]
        self.N = len(slices)
        probe = np.asarray(probe)
        if probe.ndim == 2:
            probe = probe[None]
        self.M = probe.shape[0]
        self.fny, self.fnx = int(probe.shape[-2]), int(probe.shape[-1])

        z = np.asarray(z, dtype=float).reshape(-1)
        if len(z) != self.N:
            raise ValueError("z has %d entries for %d slices" % (len(z), self.N))
        if abs(z[0]) > 0:
            z = z - z[0]
        self.z = z

        self.ob = torch.stack([torch.as_tensor(s, dtype=self.cdt, device=self.dev)
                               for s in slices])
        self.pr = torch.as_tensor(np.ascontiguousarray(probe),
                                  dtype=self.cdt, device=self.dev)
        self.ony, self.onx = int(self.ob.shape[-2]), int(self.ob.shape[-1])

        dlow = np.asarray(dlow, dtype=np.int64).reshape(-1, 2)
        ok = ((dlow[:, 0] >= 0) & (dlow[:, 1] >= 0)
              & (dlow[:, 0] + self.fny <= self.ony)
              & (dlow[:, 1] + self.fnx <= self.onx))
        if not ok.all():
            print("[warn] dropping %d of %d positions that fall outside the "
                  "object array" % (int((~ok).sum()), len(ok)))
            dlow = dlow[ok]
            if measured is not None:
                measured = np.asarray(measured)[ok]
            if mask is not None:
                mask = np.asarray(mask)[ok]
        if len(dlow) == 0:
            raise RuntimeError("no usable scan positions; check the position "
                               "convention (see positions_to_dlow)")
        self.n_pos = len(dlow)
        self.rows = torch.as_tensor(dlow[:, 0][:, None] + np.arange(self.fny),
                                    dtype=torch.long, device=self.dev)
        self.cols = torch.as_tensor(dlow[:, 1][:, None] + np.arange(self.fnx),
                                    dtype=torch.long, device=self.dev)

        self.measured = None if measured is None else torch.as_tensor(
            np.asarray(measured), dtype=self.rdt, device=self.dev)
        self.mask = None if mask is None else torch.as_tensor(
            np.asarray(mask), dtype=self.rdt, device=self.dev)

        self.prop = AngularSpectrum((self.fny, self.fnx), psize, lam,
                                    self.dev, self.cdt, self.rdt)
        self.det = farfield if farfield is not None else FarField()
        self.has_real_farfield = farfield is not None
        self.gauge = str(gauge)
        self.alpha = np.ones(self.N, dtype=complex)
        if self.gauge == 'mean':
            self._gauge_fix()

    def _coverage(self):
        cov = torch.zeros((self.ony, self.onx), dtype=torch.bool, device=self.dev)
        for s in range(0, self.n_pos, 256):
            idx = torch.arange(s, min(s + 256, self.n_pos), device=self.dev)
            r = self.rows[idx][:, :, None].expand(-1, self.fny, self.fnx)
            c = self.cols[idx][:, None, :].expand(-1, self.fny, self.fnx)
            cov[r.reshape(-1), c.reshape(-1)] = True
        return cov

    def _gauge_fix(self):
        cov = self._coverage()
        a = []
        for j in range(self.N):
            m = self.ob[j][cov].mean()
            if float(m.abs()) < 1e-9:
                m = torch.as_tensor(1.0 + 0j, dtype=self.cdt, device=self.dev)
            self.ob[j] = self.ob[j] / m
            a.append(complex(m.cpu()))
        self.alpha = np.array(a)
        c = torch.as_tensor(np.prod(self.alpha), dtype=self.cdt, device=self.dev)
        self.pr = self.pr * c

    def set_dlow(self, dlow):
        dlow = np.asarray(dlow, dtype=np.int64).reshape(-1, 2)
        ok = ((dlow[:, 0] >= 0) & (dlow[:, 1] >= 0)
              & (dlow[:, 0] + self.fny <= self.ony)
              & (dlow[:, 1] + self.fnx <= self.onx))
        dlow = dlow[ok]
        if len(dlow) == 0:
            raise RuntimeError("no usable scan positions")
        self.n_pos = len(dlow)
        self.rows = torch.as_tensor(dlow[:, 0][:, None] + np.arange(self.fny),
                                    dtype=torch.long, device=self.dev)
        self.cols = torch.as_tensor(dlow[:, 1][:, None] + np.arange(self.fnx),
                                    dtype=torch.long, device=self.dev)
        return int((~ok).sum())

    def _gather(self, arr, idx):
        return arr[self.rows[idx][:, :, None], self.cols[idx][:, None, :]]

    def run(self, batch_size=8, max_order=2, blocks=(1,), counts=None,
            n_positions=None, verbose=True):
        N = self.N
        max_order = int(min(max_order, N))
        blocks = tuple(sorted({int(b) for b in blocks if 1 <= int(b) <= N}))

        order_num = np.zeros(max_order + 1)
        order_den = np.zeros(max_order + 1)
        blk_num = {b: 0.0 for b in blocks}
        blk_den = {b: 0.0 for b in blocks}
        born_num = born_den = 0.0
        lin_num = lin_den = 0.0            # ISS with linearised transmission
        ms_lin_num = ms_lin_den = 0.0      # Exact Multislice with linearised transmission
        lvb_num = lvb_den = 0.0            
        gre_num = gabs_num = gwgt = 0.0    

        eps1_pos, phi_pos, eps1lin_pos = [], [], []
        int_stats = dict(dI_rel_num=0.0, dI_rel_den=0.0, nse=[], frac_above=[])
        example = {}

        sel = np.arange(self.n_pos)
        if n_positions is not None and n_positions < self.n_pos:
            sel = np.linspace(0, self.n_pos - 1, int(n_positions)).astype(int)

        phi = illumination(self.pr, self.prop, self.z)
        pr_int = (self.pr.abs() ** 2).sum(0)

        for s in range(0, len(sel), batch_size):
            idx = torch.as_tensor(sel[s:s + batch_size], dtype=torch.long, device=self.dev)
            O = torch.stack([self._gather(self.ob[j], idx) for j in range(N)])
            g = O - 1.0

            # --- exact multislice + the truncation hierarchy ---------------
            psi_ms, orders = order_fields(O, self.pr, self.prop, self.z, max_order)
            F_ms = self.det(psi_ms)

            cum = torch.zeros_like(orders[0])
            for m in range(max_order + 1):
                cum = cum + orders[m]
                F_m = self.det(cum)
                d = (F_ms - F_m)
                order_num[m] += float((d.abs() ** 2).sum())
                order_den[m] += float((F_m.abs() ** 2).sum())
                if m == 1:
                    per_num = (d.abs() ** 2).sum(dim=(1, 2, 3))
                    per_den = (F_m.abs() ** 2).sum(dim=(1, 2, 3)).clamp_min(1e-30)
                    eps1_pos.append(np.sqrt((per_num / per_den).cpu().numpy()))
                    F_born_h = F_m
                    if counts is not None:
                        self._intensity_stats(F_ms, F_m, counts, int_stats)
                    if not example:
                        example = self._example(F_ms, F_m, counts)

            # --- independent ISS evaluation, Eq. (model) -----------
            psi_b = born_exit(g, self.pr, phi, self.prop, self.z)
            F_b = self.det(psi_b)
            born_num += float(((F_ms - F_b).abs() ** 2).sum())
            born_den += float((F_b.abs() ** 2).sum())

            # --- linearised transmission, g_j = i s_j ----------------------
            g_lin = torch.log(O)
            F_lin = self.det(born_exit(g_lin, self.pr, phi, self.prop, self.z))
            lin_num += float(((F_ms - F_lin).abs() ** 2).sum())
            lin_den += float((F_lin.abs() ** 2).sum())
            pn = ((F_ms - F_lin).abs() ** 2).sum(dim=(1, 2, 3))
            pd = (F_lin.abs() ** 2).sum(dim=(1, 2, 3)).clamp_min(1e-30)
            eps1lin_pos.append(np.sqrt((pn / pd).cpu().numpy()))
            lvb_num += float(((F_b - F_lin).abs() ** 2).sum())
            lvb_den += float((F_b.abs() ** 2).sum())

            # --- exact multislice but with linearised slices ------------------
            # Isolates the weak-phase artifact from the multiple-scattering one
            psi_ms_lin = multislice_exit(1.0 + g_lin, self.pr, self.prop, self.z)
            F_ms_lin = self.det(psi_ms_lin)
            ms_lin_num += float(((F_ms - F_ms_lin).abs() ** 2).sum())
            ms_lin_den += float((F_ms_lin.abs() ** 2).sum())

            wgt = pr_int[None, None]
            gd = (g - g_lin)
            gre_num += float((gd.real.abs() * wgt).sum())
            gabs_num += float((g.abs() * wgt).sum())
            gwgt += float(wgt.sum()) * N * len(idx)

            # --- blocked truncation, Eq. (blocked) ------------------------
            for b in blocks:
                F_blk = self.det(blocked_exit(g, self.pr, self.prop, self.z, b))
                blk_num[b] += float(((F_ms - F_blk).abs() ** 2).sum())
                blk_den[b] += float((F_blk.abs() ** 2).sum())

            # --- probe-weighted total projected phase ---------------------
            ph = torch.stack([torch.angle(O[j]) for j in range(N)]).sum(0)
            w = pr_int[None]
            phi_pos.append(np.sqrt(
                ((ph ** 2 * w).sum(dim=(1, 2)) / w.sum().clamp_min(1e-30)
                 ).cpu().numpy()))

            del O, g, psi_ms, orders, cum, psi_b, psi_ms_lin
            if verbose:
                print("\r  %d / %d positions" % (min(s + batch_size, len(sel)),
                                                 len(sel)), end='', flush=True)
        if verbose:
            print()

        res = ScanResult()
        res.eps_order = {m: float(np.sqrt(order_num[m] / max(order_den[m], 1e-300)))
                         for m in range(max_order + 1)}
        res.eps_block = {b: float(np.sqrt(blk_num[b] / max(blk_den[b], 1e-300)))
                         for b in blocks}
        res.eps1_per_pos = np.concatenate(eps1_pos) if eps1_pos else np.array([])
        res.eps1lin_per_pos = (np.concatenate(eps1lin_pos) if eps1lin_pos
                               else np.array([]))
        res.phi_per_pos = np.concatenate(phi_pos) if phi_pos else np.array([])
        res.example = example

        eps_born = float(np.sqrt(born_num / max(born_den, 1e-300)))
        phi_rms = float(np.sqrt((res.phi_per_pos ** 2).mean())) if len(res.phi_per_pos) else float('nan')
        res.meta = dict(
            n_slices=N, n_probe_modes=self.M, n_positions=int(len(sel)),
            frame=[self.fny, self.fnx], z=list(map(float, self.z)),
            wavelength_m=self.prop.lam, psize_m=list(self.prop.psize),
            gauge=self.gauge, alpha=[complex(a).__repr__() for a in self.alpha],
            eps_born_direct=eps_born,
            eps_linearised=float(np.sqrt(lin_num / max(lin_den, 1e-300))),
            eps_multislice_linearised=float(np.sqrt(ms_lin_num / max(ms_lin_den, 1e-300))),
            eps_linearised_vs_born=float(np.sqrt(lvb_num / max(lvb_den, 1e-300))),
            g_real_gap=float(gre_num / max(gwgt, 1e-300)),
            g_mean_modulus=float(gabs_num / max(gwgt, 1e-300)),
            phi_tot_rms=phi_rms,
            eps1_estimate_correlated=float(0.5 * phi_rms ** 2 * (1 - 1.0 / N)),
            eps1_estimate_decorrelated=float(phi_rms ** 2 / (np.sqrt(2) * N)),
            na_supported=self.prop.na_max,
            shear_ac_m=float(2 * self.prop.na_max * (self.z[-1] - self.z[0]))
            if N > 1 else 0.0,
        )
        e1 = res.eps_order.get(1, eps_born)
        res.meta['counts_visible_above'] = float(1.0 / (4 * max(e1, 1e-30) ** 2))
        if counts is not None:
            res.meta['counts_per_frame'] = float(counts)
            res.meta['counts_plausible'] = bool(
                float(counts) / (self.fny * self.fnx) < 1e5)
            if int_stats['nse']:
                nse = np.concatenate(int_stats['nse'])
                res.meta['dI_over_shot_noise'] = dict(
                    median=float(np.median(nse)), p99=float(np.percentile(nse, 99)),
                    max=float(nse.max()),
                    frac_above_1=float(np.mean(np.concatenate(int_stats['frac_above']))))
            res.meta['dI_relative_l2'] = float(np.sqrt(
                int_stats['dI_rel_num'] / max(int_stats['dI_rel_den'], 1e-300)))
        return res

    def _intensity_stats(self, F_ms, F_m, counts, acc):
        I_ms = (F_ms.abs() ** 2).sum(dim=1)
        I_m = (F_m.abs() ** 2).sum(dim=1)
        scale = float(counts) / I_ms.sum(dim=(1, 2)).clamp_min(1e-30)
        I_ms = I_ms * scale[:, None, None]
        I_m = I_m * scale[:, None, None]
        dI = (I_m - I_ms).abs()
        acc['dI_rel_num'] += float((dI ** 2).sum())
        acc['dI_rel_den'] += float((I_ms ** 2).sum())
        nse = (dI / I_ms.clamp_min(1.0).sqrt()).reshape(-1)
        acc['frac_above'].append(
            np.array([float((nse > 1).to(self.rdt).mean())]))
        cap = 200000
        if nse.numel() > cap:
            sub = torch.randint(0, nse.numel(), (cap,), device=nse.device)
            nse = nse[sub]
        acc['nse'].append(nse.cpu().numpy())

    def _example(self, F_ms, F_m, counts):
        I_ms = (F_ms[0].abs() ** 2).sum(0)
        I_m = (F_m[0].abs() ** 2).sum(0)
        if counts is not None:
            sc = float(counts) / float(I_ms.sum().clamp_min(1e-30))
            I_ms, I_m = I_ms * sc, I_m * sc
        sh = (lambda a: a) if self.has_real_farfield else np.fft.fftshift
        cpp = (float(counts) / (self.fny * self.fnx)) if counts else None
        return dict(I_ms=sh(I_ms.cpu().numpy()),
                    I_born=sh(I_m.cpu().numpy()),
                    dI=sh((I_m - I_ms).abs().cpu().numpy()),
                    counts_ok=np.array(bool(cpp is not None and cpp < 1e5)))

# ===========================================================================
#  Loading
# ===========================================================================

def _h5_walk(f, want, limit=64):
    hits = []
    def visit(name, obj):
        if len(hits) >= limit:
            return None
        if isinstance(obj, h5py.Dataset) and want(name, obj):
            hits.append((name, obj))
        return None
    f.visititems(visit)
    return hits


def load_slices(fpath, verbose=True):
    if h5py is None:
        raise ImportError("h5py is required to read the reconstruction")
    out = dict(slices=[], psize=None, origin=None, thickness=None, n_slices=None)
    with h5py.File(fpath, 'r') as f:
        if 'content/objects' not in f:
            raise KeyError("%s has no content/objects group; is this the "
                           "fslices output?" % fpath)
        grp = f['content/objects']
        names = sorted(grp.keys(),
                       key=lambda s: int(''.join(c for c in s.split('_')[-1]
                                                 if c.isdigit()) or 0))
        for n in names:
            sub = grp[n]
            skeys = [k for k in sub.keys() if isinstance(sub[k], h5py.Group)]
            if not skeys:
                continue
            st = sub[skeys[0]]
            data = np.array(st['data'])
            out['slices'].append(data[0] if data.ndim == 3 else data)
            if out['psize'] is None and '_psize' in st:
                out['psize'] = np.array(st['_psize']).reshape(-1)
            if out['origin'] is None and '_origin' in st:
                out['origin'] = np.array(st['_origin']).reshape(-1)
        for key, tgt in (('content/number_of_slices', 'n_slices'),
                         ('content/slice_thickness', 'thickness')):
            if key in f:
                v = np.array(f[key])
                out[tgt] = v.reshape(-1) if v.size > 1 else float(v)
    if not out['slices']:
        raise RuntimeError("no object slices found in %s" % fpath)
    if verbose:
        print("[load] %d slices %s, psize = %s"
              % (len(out['slices']), out['slices'][0].shape, out['psize']))
    return out


def load_ptyd(fpath, want_frames=0, count_frames=8, verbose=True):
    if h5py is None:
        raise ImportError("h5py is required to read the .ptyd")
    out = dict(positions=None, frames=None, counts=None)
    with h5py.File(fpath, 'r') as f:
        if 'chunks' not in f:
            return out
        keys = sorted(f['chunks'].keys(),
                      key=lambda s: int(''.join(c for c in s if c.isdigit()) or 0))
        pos, frames, counts = [], [], []
        for k in keys:
            grp = f['chunks'][k]
            if 'positions' in grp:
                pos.append(np.array(grp['positions'], dtype=float))
            if 'data' in grp:
                d = grp['data']
                if len(frames) < want_frames:
                    take = min(want_frames - len(frames), d.shape[0])
                    frames.append(np.array(d[:take], dtype=float))
                if len(counts) < count_frames:
                    n = min(count_frames - len(counts), d.shape[0])
                    counts += [float(np.array(d[i]).sum()) for i in range(n)]
        if pos:
            out['positions'] = np.concatenate(pos, axis=0)
        if frames:
            out['frames'] = np.concatenate(frames, axis=0)
        if counts:
            out['counts'] = float(np.median(counts))
    if verbose and out['positions'] is not None:
        print("[load] %d positions from the .ptyd, median %.3g counts/frame"
              % (len(out['positions']), out['counts'] or float('nan')))
    return out


def predict_frames(scan, ks):
    idx = torch.as_tensor(list(ks), dtype=torch.long, device=scan.dev)
    O = torch.stack([scan._gather(scan.ob[j], idx) for j in range(scan.N)])
    phi = illumination(scan.pr, scan.prop, scan.z)
    I_ms = (scan.det(multislice_exit(O, scan.pr, scan.prop, scan.z)).abs() ** 2
            ).sum(dim=1).cpu().numpy()
    I_b = (scan.det(born_exit(O - 1.0, scan.pr, phi, scan.prop, scan.z)).abs() ** 2
           ).sum(dim=1).cpu().numpy()
    return I_ms, I_b


def plot_frame_comparison(scan, frames, prefix='eps', n=3, fftshift_data=False):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except Exception as e:
        print("[warn] plotting unavailable: %s" % e)
        return

    n = int(min(n, len(frames), scan.n_pos))
    if n < 1:
        return
    ks = list(range(n))
    I_ms, I_b = predict_frames(scan, ks)

    fig, ax = plt.subplots(n, 4, figsize=(11.5, 3.0 * n), squeeze=False)
    for i, k in enumerate(ks):
        meas = np.fft.fftshift(frames[k]) if fftshift_data else frames[k]
        meas = np.asarray(meas, dtype=float)
        sh = (lambda a: a) if scan.has_real_farfield else np.fft.fftshift
        ms, bn = sh(I_ms[i]), sh(I_b[i])
        # Calculate factor once using the original ms.sum()
        scale_factor = max(meas.sum(), 1e-30) / max(ms.sum(), 1e-30)
        ms = ms * scale_factor
        bn = bn * scale_factor
        top = max(meas.max(), ms.max())
        norm = LogNorm(vmin=max(top * 1e-7, 1e-12), vmax=max(top, 1e-12))
        c = _corr(ms, meas)
        panels = [(meas, 'measured (.ptyd)'),
                  (ms, 'multislice, corr %.3f' % c),
                  (bn, 'ISS, Eq. (model)')]
        for j, (img, title) in enumerate(panels):
            im = ax[i, j].imshow(np.maximum(img, 1e-12), norm=norm, cmap='inferno')
            ax[i, j].set_title(title if i == 0 else title.split(',')[0]
                               + (', corr %.3f' % c if j == 1 else ''),
                               fontsize=9)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        d = np.abs(ms - bn)
        im = ax[i, 3].imshow(np.maximum(d, 1e-12),
                             norm=LogNorm(vmin=max(top * 1e-7, 1e-12),
                                          vmax=max(top, 1e-12)),
                             cmap='inferno')
        ax[i, 3].set_title('|multislice - Born|', fontsize=9)
        ax[i, 3].set_xticks([]); ax[i, 3].set_yticks([])
        ax[i, 0].set_ylabel('frame %d' % k, fontsize=9)
        fig.colorbar(im, ax=ax[i, 3], fraction=0.046, pad=0.04)

    fig.suptitle('measured against simulated, same position and same scale',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(prefix + '_frames.png', dpi=180)
    plt.close(fig)
    print("[save] %s_frames.png" % prefix)


def check_provenance(slices, ptyr_obj):
    if ptyr_obj is None:
        return None
    prod = np.ones_like(slices[0])
    for s in slices:
        prod = prod * s
    if prod.shape != ptyr_obj.shape:
        print("[check] provenance: object shapes differ, %s vs %s -- the .ptyr "
              "is from a different run" % (prod.shape, ptyr_obj.shape))
        return 1.0
    a, b = prod.ravel(), ptyr_obj.ravel()
    alpha = (np.vdot(b, a) / max(np.vdot(b, b).real, 1e-30))
    rel = float(np.linalg.norm(a - alpha * b) / max(np.linalg.norm(a), 1e-30))
    print("[check] provenance: |prod_j O_j - obj| / |prod_j O_j| = %.3e  %s"
          % (rel, 'same run' if rel < 0.1 else
             'MISMATCH -- the probe and the slices are from different runs'))
    return rel


def build_ptypy_farfield(lam, distance, det_psize, shape, resolution,
                         device, cdtype, propagation='farfield'):
    if pu is None:
        print("[warn] ptypy not importable; falling back to an orthonormal FFT "
              "(eps_M unaffected, absolute counts indicative)")
        return None
    try:
        from ptypy.core.geometry import get_propagator
        g = pu.Param()
        g.lam = float(lam)
        g.energy = None
        g.distance = float(distance)
        g.psize = pu.expect2(det_psize)
        g.resolution = pu.expect2(resolution)
        g.shape = pu.expect2(shape)
        g.misfit = pu.expect2(0.)
        g.origin = 'fftshift'
        g.center = 'fftshift'
        g.propagation = propagation
        g.ffttype = 'scipy'
        prop = get_propagator(g)
        return FarField.from_ptypy(prop.fw, pu.expect2(shape), device, cdtype,
                                   name='ptypy %s fw' % propagation)
    except Exception as e:
        print("[warn] could not build ptypy's propagator (%s); using an "
              "orthonormal FFT" % e)
        return None


def _corr(a, b):
    a = np.sqrt(np.maximum(np.asarray(a, float), 0)).ravel()
    b = np.sqrt(np.maximum(np.asarray(b, float), 0)).ravel()
    if a.shape != b.shape:
        return float('nan')
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def _predict_frames(scan, dlow, ks):
    fy, fx = scan.fny, scan.fnx
    out = []
    for k in ks:
        r0, c0 = int(dlow[k, 0]), int(dlow[k, 1])
        if r0 < 0 or c0 < 0 or r0 + fy > scan.ony or c0 + fx > scan.onx:
            return None
        O = torch.stack([scan.ob[j][r0:r0 + fy, c0:c0 + fx][None]
                         for j in range(scan.N)])
        psi = multislice_exit(O, scan.pr, scan.prop, scan.z)
        out.append((scan.det(psi).abs() ** 2).sum(dim=1)[0].cpu().numpy())
    return np.stack(out)


def dlow_candidates(positions, origin, psize, frame):
    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    org = np.asarray(origin, dtype=float).reshape(-1)[:2]
    ps = np.asarray(psize, dtype=float).reshape(-1)
    ps = np.repeat(ps, 2)[:2]
    half = (np.asarray(frame, dtype=np.int64) // 2)
    out = {}
    for swap in (False, True):
        p = pos[:, ::-1] if swap else pos
        for sgn in (1.0, -1.0):
            pix = (sgn * p - org[None, :]) / ps[None, :]
            for corner in (False, True):
                off = np.zeros(2, np.int64) if corner else half
                key = "%s/%s/%s" % ('xy' if swap else 'yx',
                                    '+' if sgn > 0 else '-',
                                    'corner' if corner else 'centre')
                out[key] = np.round(pix).astype(np.int64) - off[None, :]
    return out


def diagnose_positions(scan, positions, origin, psize, frames, n=4):
    ks = list(range(int(min(n, len(frames), len(positions)))))
    cands = dlow_candidates(positions, origin, psize, (scan.fny, scan.fnx))
    print("[check] position convention scan (correlation with %d measured "
          "frames)" % len(ks))
    print("        %-16s %8s %8s" % ('convention', 'as-is', 'fftshift'))
    best = (None, None, -2.0, False)
    for name, dl in cands.items():
        pred = _predict_frames(scan, dl, ks)
        if pred is None:
            print("        %-16s %8s %8s" % (name, 'oob', 'oob'))
            continue
        c0 = float(np.mean([_corr(pred[i], frames[i]) for i in ks]))
        c1 = float(np.mean([_corr(pred[i], np.fft.fftshift(frames[i]))
                            for i in ks]))
        print("        %-16s %8.3f %8.3f" % (name, c0, c1))
        if max(c0, c1) > best[2]:
            best = (name, dl, max(c0, c1), c1 > c0)
    if best[0] is None:
        print("        every candidate falls outside the object array")
        return best
    print("        best: %s%s, correlation %.3f"
          % (best[0], ' with the frames fftshifted' if best[3] else '', best[2]))
    if best[3]:
        print("        note: the stored frames appear fft-ordered relative to "
              "the propagator output; this affects the counts test only, not "
              "eps_M.")
    return best


def check_positions(scan, frames, n=4):
    n = int(min(n, len(frames), scan.n_pos))
    if n < 1:
        return None
    idx = torch.arange(n, dtype=torch.long, device=scan.dev)
    O = torch.stack([scan._gather(scan.ob[j], idx) for j in range(scan.N)])
    I = (scan.det(multislice_exit(O, scan.pr, scan.prop, scan.z)).abs() ** 2
         ).sum(dim=1).cpu().numpy()
    cs = []
    for k in range(n):
        a = np.sqrt(np.maximum(I[k], 0)).ravel()
        b = np.sqrt(np.maximum(frames[k], 0)).ravel()
        a = a - a.mean()
        b = b - b.mean()
        cs.append(float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)))
    c = float(np.mean(cs))
    print("[check] amplitude correlation with %d measured frames: %.3f %s"
          % (n, c, 'OK' if c > 0.7 else
             'LOW -- the position convention is probably wrong'))
    return c


def load_ptyr(fpath, verbose=True):
    if h5py is None:
        raise ImportError("h5py is required to read the .ptyr")
    out = dict(probe=None, pr_psize=None, ob_origin=None, ob_psize=None,
               obj=None, positions=None)
    with h5py.File(fpath, 'r') as f:
        pr = _h5_walk(f, lambda n, d: n.endswith('/data') and '/probe/' in n)
        if pr:
            out['probe'] = np.array(pr[0][1])
            base = pr[0][0].rsplit('/', 1)[0]
            if base + '/_psize' in f:
                out['pr_psize'] = np.array(f[base + '/_psize']).reshape(-1)
        ob = _h5_walk(f, lambda n, d: n.endswith('/data')
                      and ('/obj/' in n or '/object/' in n))
        if ob:
            base = ob[0][0].rsplit('/', 1)[0]
            d = np.array(ob[0][1])
            out['obj'] = d[0] if d.ndim == 3 else d
            if base + '/_origin' in f:
                out['ob_origin'] = np.array(f[base + '/_origin']).reshape(-1)
            if base + '/_psize' in f:
                out['ob_psize'] = np.array(f[base + '/_psize']).reshape(-1)
        pos = _h5_walk(f, lambda n, d: 'pos' in n.lower() and d.ndim == 2
                       and d.shape[-1] == 2 and d.shape[0] > 4)
        if pos:
            pos.sort(key=lambda kv: -kv[1].shape[0])
            out['positions'] = np.array(pos[0][1], dtype=float)
            if verbose:
                print("[load] positions from '%s', %d points"
                      % (pos[0][0], len(out['positions'])))
    if verbose and out['probe'] is not None:
        print("[load] probe %s" % (out['probe'].shape,))
    return out


def positions_to_dlow(coords_m, origin_m, psize, frame_shape):
    coords_m = np.asarray(coords_m, dtype=float).reshape(-1, 2)
    origin_m = np.asarray(origin_m, dtype=float).reshape(-1)
    psize = np.asarray(psize, dtype=float).reshape(-1)
    if psize.size == 1:
        psize = np.repeat(psize, 2)
    pix = (coords_m - origin_m[None, :2]) / psize[None, :2]
    dcoord = np.round(pix).astype(np.int64)
    return dcoord - np.asarray(frame_shape, dtype=np.int64)[None, :] // 2


def raster_dlow(object_shape, frame_shape, step_pix, border=0):
    oy, ox = int(object_shape[0]), int(object_shape[1])
    fy, fx = int(frame_shape[0]), int(frame_shape[1])
    step = int(max(1, step_pix))
    ys = np.arange(border, max(border + 1, oy - fy - border + 1), step)
    xs = np.arange(border, max(border + 1, ox - fx - border + 1), step)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    return np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.int64)


# ===========================================================================
#  In-session entry point (exact positions, exact data)
# ===========================================================================

def epsilon_from_ptycho(P, slices_h5=None, use_measured=True, **kw):
    di = getattr(P, 'diff', None) or getattr(P, 'di')
    prc = getattr(P, 'probe', None) or getattr(P, 'pr')
    obc = getattr(P, 'obj', None) or getattr(P, 'ob')

    geom = list(P.model.scans.values())[0].geometries[0]
    if getattr(geom, 'resample', 1) not in (1, None):
        print("[warn] geometry resample = %s; both models are evaluated on the "
              "reconstruction grid, so the detector-plane counts are indicative"
              % geom.resample)
    prs = list(prc.storages.values())[0]
    probe = np.ascontiguousarray(prs.data)

    if slices_h5 is not None:
        sl = load_slices(slices_h5)
        slices = sl['slices']
        thickness = sl['thickness']
    else:
        obs = list(obc.storages.values())[0]
        if obs.data.shape[0] != 1:
            raise NotImplementedError("single object mode only, as in the engine")
        slices = [obs.data[0]]
        thickness = 0.0

    N = len(slices)
    th = kw.pop('thickness', thickness)
    z = _z_from_thickness(th, N)

    names = sorted(n for n, v in di.views.items() if v.active)
    if not names:
        raise RuntimeError("no active diffraction views; build P at level >= 4")
    dlow, meas, mask = [], [], []
    fw = None
    for n in names:
        v = di.views[n]
        pods = list(v.pods.values())
        pod = pods[0]
        if fw is None:
            fw = pod.fw
        dlow.append(np.asarray(pod.ob_view.dlow, dtype=np.int64))
        if use_measured:
            meas.append(np.asarray(v.data, dtype=float))
            m = pod.mask
            mask.append(np.ones_like(meas[-1]) if m is None
                        else np.asarray(m, dtype=float))
    dlow = np.stack(dlow)
    measured = np.stack(meas) if meas else None
    mask = np.stack(mask) if mask else None
    print("[ptypy] %d active views, %d probe mode(s), frame %s"
          % (len(names), probe.shape[0], tuple(probe.shape[-2:])))

    plot = kw.pop('plot', False)
    out = kw.pop('out', None)
    run_kw = {k: kw.pop(k) for k in
              ('batch_size', 'max_order', 'blocks', 'counts', 'n_positions',
               'verbose') if k in kw}
    if measured is not None and 'counts' not in run_kw:
        run_kw['counts'] = float(np.median(measured.sum(axis=(1, 2))))

    device = kw.get('device', 'cuda')
    precision = kw.get('precision', 'single')
    dev = torch.device(device if not (str(device).startswith('cuda')
                                      and not torch.cuda.is_available()) else 'cpu')
    cdt = torch.complex128 if str(precision).startswith('d') else torch.complex64
    ff = FarField.from_ptypy(fw, probe.shape[-2:], dev, cdt) if fw is not None else None

    scan = BornErrorScan(slices, probe, dlow,
                         psize=geom.resolution, lam=geom.lam, z=z,
                         farfield=ff, **kw)
    res = scan.run(**run_kw)
    _report(res)
    if out:
        _save(res, out)
    if plot:
        make_plots(res, out or 'eps')
    return res


def _z_from_thickness(th, N):
    if th is None:
        th = 0.0
    if np.ndim(th) and np.size(th) > 1:
        d = np.asarray(th, dtype=float).reshape(-1)
        if len(d) == N - 1:
            return np.concatenate([[0.0], np.cumsum(d)])
        if len(d) == N:
            return np.concatenate([[0.0], np.cumsum(d[:-1])])
        raise ValueError("slice_thickness has %d entries for %d slices"
                         % (len(d), N))
    return np.arange(N, dtype=float) * float(np.ravel(th)[0] if np.ndim(th) else th)


# ===========================================================================
#  Reporting
# ===========================================================================

def _report(res: ScanResult):
    m = res.meta
    N = m['n_slices']
    print()
    print("=" * 68)
    print("ISS truncation error, Eq. (epsilon)")
    print("=" * 68)
    print("slices              : %d   z = %s (um)"
          % (N, np.round(np.array(m['z']) * 1e6, 4).tolist()))
    print("positions / modes   : %d / %d" % (m['n_positions'], m['n_probe_modes']))
    print("gauge               : %s" % m['gauge'])
    print()
    print("  M    eps_M         interpretation")
    for M, e in sorted(res.eps_order.items()):
        tag = ('unscattered only' if M == 0 else
               'Eq. (model), ISS' if M == 1 else
               'exact multislice' if M == N else 'order <= %d' % M)
        print("  %-4d %-13.4e %s" % (M, e, tag))
    print("  direct Born evaluation: %.4e  (should match M = 1)"
          % m['eps_born_direct'])
    if res.eps_block:
        print()
        print("  blocks   eps          Eq. (blocked) predicts ~ sqrt(1 - 1/M) "
              "of the M=1 value")
        e1 = res.eps_block.get(1, res.eps_order.get(1, np.nan))
        for b, e in sorted(res.eps_block.items()):
            if b >= N:
                print("  %-8d %-12.4e exact multislice (M = N), so ~0 by "
                      "construction" % (b, e))
            else:
                print("  %-8d %-12.4e ratio to 1 block = %.3f  (predicted %.3f)"
                      % (b, e, e / max(e1, 1e-30),
                         np.sqrt(max(1 - 1.0 / b, 0.0)) if b > 1 else 1.0))

    print()
    print("Perturbation choice (Sec. 'The perturbation'):")
    print("  Born, exact transmission g = O - 1     eps_1 = %.4e"
          % res.eps_order.get(1, m['eps_born_direct']))
    print("  Born, tangent linearisation g = i s    eps_1 = %.4e"
          % m['eps_linearised'])
    print("  Multislice, tangent linearisation      eps   = %.4e"
          % m['eps_multislice_linearised'])
    ratio = m['eps_linearised'] / max(res.eps_order.get(1, m['eps_born_direct']), 1e-30)
    print("  linearising alone costs a factor %.2f, and contributes %.4e"
          % (ratio, m['eps_linearised_vs_born']))
    print("  relative to the exact-g Born field. Object domain: the chord and")
    print("  the tangent differ by <|Re(g - i s)|> = %.4e against <|g|> = %.4e"
          % (m['g_real_gap'], m['g_mean_modulus']))
    print("  (probe weighted). That gap is the O(phi^2) real part; it does not")
    print("  vanish at N = 1, where the chord model is exact.")
    print()
    print("Phi_tot (probe weighted rms) : %.4f rad" % m['phi_tot_rms'])
    print("  Eq. (eps1-estimate)  correlated : %.4e" % m['eps1_estimate_correlated'])
    print("  Eq. (eps1-decorr)  decorrelated : %.4e" % m['eps1_estimate_decorrelated'])
    print("  measured eps_1                  : %.4e"
          % res.eps_order.get(1, m['eps_born_direct']))
    print()
    print("Dose criterion, Eq. (noisecrit): the truncation is invisible while")
    print("  eps <= 1 / (2 sqrt(I)); at this eps_1 that means up to")
    print("  I ~ %.3g counts in the brightest pixel." % m['counts_visible_above'])
    if 'dI_over_shot_noise' in m:
        d = m['dI_over_shot_noise']
        cpp = m['counts_per_frame'] / float(np.prod(m['frame']))
        print("  at %.3g counts/frame (%.3g per pixel): |dI|/sqrt(I) median "
              "%.3f, p99 %.3f, max %.3f; %.2f%% of pixels above 1"
              % (m['counts_per_frame'], cpp, d['median'], d['p99'], d['max'],
                 100 * d['frac_above_1']))
        if cpp > 1e5:
            print("  [warn] %.3g counts per pixel is not a plausible photon "
                  "number. If the .ptyd is not in photon units this test is "
                  "meaningless; pass --counts with the real photon budget."
                  % cpp)
    if N > 1:
        print()
        print("Depth-separability scale, Eq. (depthres): a_c = 2 NA dz = %.3g m"
              % m['shear_ac_m'])
    print("=" * 68)


def _save(res: ScanResult, prefix):
    with open(prefix + '.json', 'w') as fh:
        json.dump(res.as_dict(), fh, indent=2)
    np.savez_compressed(
        prefix + '.npz',
        eps_order_M=np.array(sorted(res.eps_order)),
        eps_order=np.array([res.eps_order[k] for k in sorted(res.eps_order)]),
        eps_block_n=np.array(sorted(res.eps_block)),
        eps_block=np.array([res.eps_block[k] for k in sorted(res.eps_block)]),
        eps1_per_pos=res.eps1_per_pos, phi_per_pos=res.phi_per_pos,
        eps1lin_per_pos=(res.eps1lin_per_pos
                         if res.eps1lin_per_pos is not None else np.array([])),
        **{k: v for k, v in res.example.items()})
    print("[save] %s.json, %s.npz" % (prefix, prefix))


def make_plots(res: ScanResult, prefix='eps'):
    try:
        _make_plots(res, prefix)
    except Exception as e:                                   # noqa: BLE001
        print("[warn] plotting failed (%s); results are in %s.json/.npz"
              % (e, prefix))


def _make_plots(res: ScanResult, prefix='eps'):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except Exception as e:                                   # pragma: no cover
        print("[warn] plotting unavailable: %s" % e)
        return

    m = res.meta
    N = m['n_slices']
    fig, ax = plt.subplots(2, 2, figsize=(9.5, 7.5))

    a = ax[0, 0]
    Ms = sorted(res.eps_order)
    a.semilogy(Ms, [res.eps_order[k] for k in Ms], 'o-',
               label='order truncation, $g=O-1$')
    if res.eps_block and len(res.eps_block) > 1:
        bs = sorted(res.eps_block)
        a.semilogy(bs, [res.eps_block[b] for b in bs], 's--',
                   label='blocked, $g=O-1$')
    if m.get('eps_linearised'):
        a.semilogy([1], [m['eps_linearised']], 'X', ms=10, color='crimson',
                   label=r'ISS, tangent $g=is$')
    if m.get('eps_multislice_linearised'):
        a.semilogy([N], [m['eps_multislice_linearised']], '*', ms=10, color='purple',
                   label=r'multislice, tangent $g=is$')
    if 'counts_per_frame' in m and m.get('counts_plausible', True):
        npix = np.prod(m['frame'])
        floor = 1.0 / (2 * np.sqrt(max(m['counts_per_frame'] / npix, 1e-30)))
        a.axhline(floor, color='r', ls=':', label='shot-noise floor, Eq. (noisecrit)')
    a.set_xlabel('truncation order $M$ / number of blocks')
    a.set_ylabel(r'$\epsilon_M$')
    a.set_title('truncation error, $N=%d$' % N)
    if N <= 2:
        a.text(0.03, 0.06, 'at $N=2$ the two schemes coincide',
               transform=a.transAxes, fontsize=7, color='0.35')
    a.set_xticks(sorted(set(Ms) | set(res.eps_block)))
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[0, 1]
    if len(res.phi_per_pos) and len(res.eps1_per_pos):
        n = min(len(res.phi_per_pos), len(res.eps1_per_pos))
        p, e = res.phi_per_pos[:n], res.eps1_per_pos[:n]
        a.loglog(p, e, '.', ms=3, alpha=0.4, label=r'per position, $g=O-1$')
        el = res.eps1lin_per_pos
        if el is not None and len(el) >= n:
            a.loglog(p, el[:n], '.', ms=3, alpha=0.4, color='crimson',
                     label=r'per position, $g=is$')
        pp = np.linspace(max(p.min(), 1e-4), p.max(), 64)
        a.loglog(pp, 0.5 * pp ** 2 * (1 - 1.0 / N), 'k-',
                 label=r'$\frac{1}{2}\Phi^2(1-1/N)$')
        a.loglog(pp, pp ** 2 / (np.sqrt(2) * N), 'k--',
                 label=r'$\Phi^2/(\sqrt{2}N)$')
        a.set_xlabel(r'$\Phi_{\rm tot}$ (rad)')
        a.set_ylabel(r'$\epsilon_1$')
        a.set_title('which branch of Eq. (eps1-decorr)?')
        a.legend(fontsize=7)
        a.grid(alpha=0.3, which='both')

    ex = res.example
    if ex:
        a = ax[1, 0]
        I = np.maximum(ex['I_ms'], 1e-12)
        a.imshow(I, norm=LogNorm(vmin=max(I.max() * 1e-8, 1e-12), vmax=I.max()),
                 cmap='inferno')
        a.set_title('multislice intensity (one frame)')
        a.set_xticks([]); a.set_yticks([])

        a = ax[1, 1]
        ok = bool(np.asarray(ex.get('counts_ok', False)))
        if ok:
            im = a.imshow(ex['dI'] / np.sqrt(np.maximum(ex['I_ms'], 1.0)),
                          cmap='viridis')
            a.set_title(r'$|\Delta I|/\sqrt{I}$  (shot-noise units)')
        else:
            r = ex['dI'] / max(ex['I_ms'].max(), 1e-30)
            im = a.imshow(r, cmap='viridis',
                          norm=LogNorm(vmin=max(r.max() * 1e-6, 1e-12),
                                       vmax=max(r.max(), 1e-12)))
            a.set_title(r'$|\Delta I| / \max I$   (no photon calibration)')
        a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    fig.tight_layout()
    try:
        fig.savefig(prefix + '_born_error.png', dpi=200)
        print("[save] %s_born_error.png" % prefix)
    except Exception as e:                                   # noqa: BLE001
        print("[warn] could not save the figure: %s" % e)
    plt.close(fig)


# ===========================================================================
#  Self-test
# ===========================================================================

def selftest(device='cpu', seed=0):
    if torch is None:
        raise ImportError("selftest requires PyTorch")
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    cdt = torch.complex128
    ny = nx = 64
    lam, dx, dz = 1.5e-10, 20e-9, 1.0e-3

    def make(N, phi_scale):
        yy, xx = np.mgrid[:ny, :nx] - ny / 2
        P = np.exp(-(yy ** 2 + xx ** 2) / (2 * (ny / 8) ** 2)) \
            * np.exp(1j * 0.4 * xx / nx)
        P = torch.as_tensor(P[None], dtype=cdt, device=dev)
        ob = []
        for _ in range(N):
            ph = phi_scale * (rng.random((ny, nx)) - 0.5)
            ob.append(torch.as_tensor(np.exp(1j * ph)[None], dtype=cdt, device=dev))
        O = torch.stack(ob)                                  # (N, 1, ny, nx)
        prop = AngularSpectrum((ny, nx), (dx, dx), lam, dev, cdt, torch.float64)
        z = np.arange(N) * dz
        return O, P, prop, z

    def rel(a, b):
        return float((a - b).abs().max()) / max(float(b.abs().max()), 1e-300)

    ok = True

    O, P, prop, z = make(1, 0.6)
    psi_ms = multislice_exit(O, P, prop, z)
    psi_b = born_exit(O - 1.0, P, illumination(P, prop, z), prop, z)
    e = rel(psi_b, psi_ms)
    print("1. N=1 exactness                 : %.3e %s" % (e, 'OK' if e < 1e-12 else 'FAIL'))
    ok &= e < 1e-12

    O, P, prop, z = make(4, 0.5)
    psi_ms, orders = order_fields(O, P, prop, z, max_order=4)
    e = rel(orders.sum(0), psi_ms)
    print("2. sum of orders == multislice   : %.3e %s" % (e, 'OK' if e < 1e-10 else 'FAIL'))
    ok &= e < 1e-10

    psi_b = born_exit(O - 1.0, P, illumination(P, prop, z), prop, z)
    e = rel(orders[0] + orders[1], psi_b)
    print("3. order<=1 == Eq. (model)       : %.3e %s" % (e, 'OK' if e < 1e-10 else 'FAIL'))
    ok &= e < 1e-10

    e = rel(blocked_exit(O - 1.0, P, prop, z, len(z)), psi_ms)
    print("4. blocked, M=N == multislice    : %.3e %s" % (e, 'OK' if e < 1e-10 else 'FAIL'))
    ok &= e < 1e-10

    print("5. eps_1 vs Phi^2 (N=4):")
    prev = None
    for scale in (0.02, 0.04, 0.08):
        O, P, prop, z = make(4, scale)
        psi_ms = multislice_exit(O, P, prop, z)
        psi_b = born_exit(O - 1.0, P, illumination(P, prop, z), prop, z)
        Fm, Fb = detector(psi_ms), detector(psi_b)
        eps = float(((Fm - Fb).abs() ** 2).sum().sqrt()
                    / (Fb.abs() ** 2).sum().sqrt())
        r = '' if prev is None else '  ratio %.2f (expect ~4)' % (eps / prev)
        print("    phase scale %.3f -> eps_1 = %.3e%s" % (scale, eps, r))
        prev = eps

    print("SELF-TEST %s" % ('PASSED' if ok else 'FAILED'))
    return ok


# ===========================================================================
#  CLI
# ===========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Measure the ISS truncation error eps_M against '
                    'exact multislice for a ptychographic reconstruction. '
                    'Defaults are the NanoMAX two-slice geometry of the '
                    'offline reconstruction script (scan 434, crop 512, '
                    '4.18 m, 1.58 mm slice separation), so with no arguments '
                    'it points at the files that script wrote.')
    ap.add_argument('--selftest', action='store_true')

    g = ap.add_argument_group('experiment (defaults from the reconstruction script)')
    g.add_argument('--basedir', default=NANOMAX['basedir'])
    g.add_argument('--sample', default=NANOMAX['sample'])
    g.add_argument('--scannr', type=int, default=NANOMAX['scannr'])
    g.add_argument('--cropping', type=int, default=NANOMAX['cropping'],
                   help='frame size, p.scans.scan00.data.shape')
    g.add_argument('--distance', type=float, default=NANOMAX['distance_m'],
                   help='sample-detector distance in m')
    g.add_argument('--det-psize', type=float, default=NANOMAX['det_psize'],
                   help='detector pixel size in m (eiger4m: 75e-6)')
    g.add_argument('--thickness', type=float, default=NANOMAX['slice_thickness'],
                   help='slice separation in m')
    g.add_argument('--n-slices', type=int, default=NANOMAX['number_of_slices'])
    g.add_argument('--iter', type=int, default=NANOMAX['niter'],
                   help='iteration number in the fslices filename')

    g = ap.add_argument_group('files (constructed from the above if omitted)')
    g.add_argument('--slices', help='engine fslices output (.h5)')
    g.add_argument('--ptyr', default='auto',
                   help="ptypy .ptyr dump for the probe and positions; 'auto' "
                        "takes the newest in rec/, then dumps/")
    g.add_argument('--ptyd', default='auto',
                   help="ptypy .ptyd data file, which is where the scan "
                        "positions actually live; 'auto' uses data/ under the "
                        "run directory. Pass '' to skip.")
    g.add_argument('--engine', default=None,
                   help="restrict the 'auto' .ptyr search to one engine, e.g. "
                        "MomentumFirstBorn. Use the engine that wrote the "
                        "slice file: the probe must come from the same run.")
    g.add_argument('--ptyr-pick', choices=('iterations', 'mtime'),
                   default='iterations',
                   help="how 'auto' breaks ties: highest iteration count in "
                        "the filename (default) or newest on disk")
    g.add_argument('--check-frames', type=int, default=4,
                   help='measured frames read from the .ptyd to check the '
                        'position convention against the data')
    g.add_argument('--no-auto-positions', action='store_true',
                   help='do not adopt a better-correlating position '
                        'convention if the default one fails the check')
    g.add_argument('--compare-frames', type=int, default=3,
                   help='frames to plot measured beside simulated in '
                        '<out>_frames.png; 0 turns it off')
    g.add_argument('--probe-npy', help='probe as .npy, (M,ny,nx) or (ny,nx)')
    g.add_argument('--positions-npy', help='positions as .npy, (n,2)')
    g.add_argument('--positions-units', choices=('m', 'pix', 'dlow'), default='m',
                   help="'m': view centres in metres (ptypy convention); "
                        "'pix': centres in object pixels; 'dlow': frame corners")
    g.add_argument('--scan', choices=('positions', 'raster'), default='positions')
    g.add_argument('--raster-step', type=int, default=32,
                   help='raster step in object pixels')

    g = ap.add_argument_group('wavelength (derived from the geometry by default)')
    g.add_argument('--energy', type=float, default=None, help='photon energy in keV')
    g.add_argument('--electron-kv', type=float, default=None)
    g.add_argument('--lam', type=float, default=None, help='wavelength in m')
    g.add_argument('--psize', type=float, default=None,
                   help='sample-plane pixel size in m (overrides the h5)')

    g = ap.add_argument_group('measurement')
    g.add_argument('--max-order', type=int, default=2,
                   help='highest truncation order; at N=2 this reaches exact '
                        'multislice, so eps_2 should come back at ~1e-7')
    g.add_argument('--blocks', default='1,2',
                   help='comma-separated block counts, e.g. 1,2,4')
    g.add_argument('--counts', type=float, default=None,
                   help='total photons per frame, for Eq. (noisecrit)')
    g.add_argument('--gauge', choices=('none', 'mean'), default='none')
    g.add_argument('--batch-size', type=int, default=16,
                   help='frames per batch; peak memory is (max_order+2) x '
                        'batch x probe modes x frame')
    g.add_argument('--n-positions', type=int, default=None)
    g.add_argument('--device', default='cuda')
    g.add_argument('--precision', choices=('single', 'double'), default='single')
    g.add_argument('--out', default='eps')
    g.add_argument('--plot', action='store_true')
    a = ap.parse_args(argv)

    if a.selftest:
        return 0 if selftest(device='cpu') else 1

    if pu is None:
        print("[warn] ptypy could not be imported (%s: %s)."
              % (type(_PTYPY_ERROR).__name__, _PTYPY_ERROR))
        print("       Falling back to an equivalent numpy grid and an "
              "orthonormal FFT. eps_M is unaffected -- it is a ratio of norms "
              "and invariant under any propagator unitary up to a scalar -- "
              "but the measured-frame comparison and the absolute counts test "
              "need ptypy's own far field. If this is an mpi4py/libmpi error, "
              "activate the environment the reconstruction ran in.")

    paths = default_paths(a.basedir, a.sample, a.scannr, a.cropping,
                          a.distance, a.thickness, a.n_slices, a.iter)
    slices_path = a.slices or paths['fslices']
    if not os.path.exists(slices_path):
        ap.error('slice file not found: %s\n(pass --slices, or adjust '
                 '--scannr/--cropping/--thickness/--iter)' % slices_path)
    print("[paths] slices : %s" % slices_path)

    ptyr_path = a.ptyr
    if ptyr_path == 'auto':
        ptyr_path = (pick_ptyr(paths['ptyr_glob'], a.engine, a.ptyr_pick)
                     or pick_ptyr(paths['dumps_glob'], a.engine, a.ptyr_pick))
        print("[paths] ptyr   : %s" % ptyr_path)

    sl = load_slices(slices_path)
    slices = sl['slices']
    N = len(slices)
    psize = a.psize if a.psize else (sl['psize'] if sl['psize'] is not None else None)
    if psize is None:
        ap.error('pixel size not found in the h5; pass --psize')

    th = a.thickness if a.thickness is not None else sl['thickness']
    if th is None:
        ap.error('slice thickness not found in the h5; pass --thickness')
    z = _z_from_thickness(th, N)

    if a.lam:
        lam = a.lam
    elif a.energy:
        lam = wavelength_from_photon_kev(a.energy)
    elif a.electron_kv:
        lam = wavelength_from_electron_kv(a.electron_kv)
    else:
        lam = wavelength_from_geometry(psize, a.cropping, a.det_psize, a.distance)
        print("[geom] lam = %.4f A from resolution %.3f nm, %d px, %.0f um "
              "detector pixel, %.3f m  ->  %.3f keV"
              % (lam * 1e10, np.ravel(psize)[0] * 1e9, a.cropping,
                 a.det_psize * 1e6, a.distance, 1.23984193e-9 / lam))

    probe = positions = None
    ptyr = load_ptyr(ptyr_path) if ptyr_path else {}
    if a.probe_npy:
        probe = np.load(a.probe_npy)
    elif ptyr.get('probe') is not None:
        probe = ptyr['probe']
    else:
        ap.error('no probe: pass --ptyr or --probe-npy')

    frame = probe.shape[-2:]
    if ptyr.get('obj') is not None:
        check_provenance(slices, ptyr['obj'])

    # Positions: the .ptyd carries them explicitly, the .ptyr usually does not.
    ptyd_path = a.ptyd
    if ptyd_path == 'auto':
        ptyd_path = paths['ptyd'] if os.path.exists(paths['ptyd']) else None
        if ptyd_path:
            print("[paths] ptyd   : %s" % ptyd_path)
    ptyd = load_ptyd(ptyd_path, want_frames=max(a.check_frames,
                                                a.compare_frames)) if ptyd_path else {}

    units = a.positions_units
    if a.positions_npy:
        positions = np.load(a.positions_npy)
    elif ptyd.get('positions') is not None:
        positions, units = ptyd['positions'], 'm'
    elif ptyr.get('positions') is not None:
        positions = ptyr['positions']

    counts = a.counts if a.counts is not None else ptyd.get('counts')

    if a.scan == 'raster' or positions is None:
        if positions is None and a.scan != 'raster':
            print("[warn] no positions found; falling back to a synthetic "
                  "raster. This covers the unconstrained border of the object, "
                  "where g = 0, and will bias eps_1 low -- point --ptyd at the "
                  "data file instead.")
        dlow = raster_dlow(slices[0].shape, frame, a.raster_step)
    elif units == 'dlow':
        dlow = np.asarray(positions, dtype=np.int64)
    elif units == 'pix':
        dlow = (np.round(positions).astype(np.int64)
                - np.asarray(frame, dtype=np.int64)[None, :] // 2)
    else:
        origin = ptyr.get('ob_origin')
        if origin is None:
            origin = sl['origin']
        if origin is None:
            ap.error('object origin not found; use --positions-units pix/dlow '
                     'or --scan raster')
        dlow = positions_to_dlow(positions, origin, psize, frame)
    print("[scan] %d frames, dlow rows %d..%d, cols %d..%d, object %s"
          % (len(dlow), dlow[:, 0].min(), dlow[:, 0].max(),
             dlow[:, 1].min(), dlow[:, 1].max(), slices[0].shape))

    dev = torch.device(a.device if not (str(a.device).startswith('cuda')
                                        and not torch.cuda.is_available()) else 'cpu')
    cdt = torch.complex128 if a.precision.startswith('d') else torch.complex64
    ff = build_ptypy_farfield(lam, a.distance, a.det_psize, frame, psize, dev, cdt)

    scan = BornErrorScan(slices, probe, dlow, psize=psize, lam=lam, z=z,
                         device=a.device, precision=a.precision, gauge=a.gauge,
                         farfield=ff)
    if ptyd.get('frames') is not None:
        c = check_positions(scan, ptyd['frames'])
        if c is not None and c < 0.7 and positions is not None and units == 'm':
            org = ptyr.get('ob_origin')
            if org is None:
                org = sl['origin']
            if org is not None:
                name, dl, cbest, shifted = diagnose_positions(
                    scan, positions, org, psize, ptyd['frames'])
                if dl is not None and cbest > max(c + 0.1, 0.5) and not a.no_auto_positions:
                    dropped = scan.set_dlow(dl)
                    print("[check] adopting the '%s' convention (%d frames, "
                          "%d dropped). Pass --no-auto-positions to keep the "
                          "default." % (name, scan.n_pos, dropped))
                else:
                    print("[warn] no convention correlates well; eps_M below "
                          "is computed on frames that may not match the data. "
                          "Use --scan raster or supply --positions-npy.")
    if ptyd.get('frames') is not None and a.compare_frames > 0:
        plot_frame_comparison(scan, ptyd['frames'], a.out, n=a.compare_frames)
    blocks = tuple(int(b) for b in str(a.blocks).split(',') if b.strip())
    res = scan.run(batch_size=a.batch_size, max_order=a.max_order,
                   blocks=blocks, counts=counts, n_positions=a.n_positions)
    _report(res)
    _save(res, a.out)
    if a.plot:
        make_plots(res, a.out)
    return 0


def epsilon_default(**kw):
    """Notebook entry point: run the file-based route with the NanoMAX
    defaults, overriding any CLI option by keyword.

        epsilon_default(iter=20, plot=True, out='eps_434')
        epsilon_default(scannr=435, thickness=1.0e-3, gauge='mean')

    Keys match the CLI flags with dashes replaced by underscores.
    """
    argv = []
    for k, v in kw.items():
        flag = '--' + k.replace('_', '-')
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv += [flag, str(v)]
    return main(argv)


if __name__ == '__main__':
    sys.exit(main())