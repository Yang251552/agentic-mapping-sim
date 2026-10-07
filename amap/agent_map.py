"""Stage 3: the agent's own map, built from Observations only (truth firewall: no backend, no ground truth).

Occupancy: int8 (n, n) grid of UNKNOWN / FREE / OCC by majority of scan evidence; executor bumps are permanent. Objects: one track per sensor instance key,
exposed only through agent ids 0, 1, 2, ... in first-detection order (the sensor key never leaves this class).
Class posterior: Bayesian fusion with the confusion likelihood L(label | c) = p if c == label else (1-p)/(n_cat-1),
p = p_bins[bin]; each (instance, bin) is fused at most once because the shared noise model draws one label per
(instance, bin) and repeating it would fake evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from amap.contracts import FREE, OCC, UNKNOWN, GridSpec, Observation


@dataclass
class Track:
    id: int
    rc: tuple[int, int]
    logpost: np.ndarray                      # (n_cat,) normalised log-posterior
    bins_seen: set[int] = field(default_factory=set)
    obs: list[tuple[int, int]] = field(default_factory=list)   # (bin, label), fused observations only


class AgentMap:
    def __init__(self, grid: GridSpec, n_cat: int, p_bins, tau: float, prior=None, kernel=None):
        """prior / kernel None: uniform prior and uniform wrong labels. Otherwise the train-split class prior pi and the
        wrong-label kernel K (row: true class) that the sensor uses: L_b(label | c) = p_b [c == label] + (1 - p_b) K[c, label]."""
        if n_cat < 2:
            raise ValueError("n_cat must be >= 2")
        if (prior is None) != (kernel is None):
            raise ValueError("prior and kernel go together")
        self.grid = grid
        self.n_cat = int(n_cat)
        self.p_bins = tuple(float(p) for p in p_bins)
        self.tau = float(tau)
        self.occ = np.full((grid.n, grid.n), UNKNOWN, dtype=np.int8)
        # Occupancy by majority vote of scan evidence (seen occupied vs seen free); an executor bump is permanent.
        # A purely sticky OCC let one grazing door-frame point seal a doorway for good (M run 10-06 19:00).
        self._hit = np.zeros((grid.n, grid.n), dtype=np.int32)
        self._miss = np.zeros((grid.n, grid.n), dtype=np.int32)
        # Livelock guards, from the agent's own history only: frontier cells still open after a look-around from
        # next to them (unobservable), and objects already given a close look from a revisit viewpoint.
        self.dead = np.zeros((grid.n, grid.n), dtype=bool)
        self.tried: set[int] = set()
        self._tracks: list[Track] = []
        self._id_of: dict[int, int] = {}     # sensor key -> agent id; private
        # per bin: (log L for c != label as an (n_cat,) row, log L for c == label)
        self._loglik = []
        for p in self.p_bins:
            off = np.full(self.n_cat, np.log((1.0 - p) / (self.n_cat - 1)))
            self._loglik.append((off, np.log(p)))
        self._logprior = np.full(self.n_cat, -np.log(self.n_cat))
        self._loglik_full = None   # kernel model: per bin, (n_cat true, n_cat label) log likelihood; read a column per label
        if kernel is not None:
            pi = np.asarray(prior, dtype=np.float64)
            K = np.asarray(kernel, dtype=np.float64)
            if pi.shape != (self.n_cat,) or K.shape != (self.n_cat, self.n_cat):
                raise ValueError("prior / kernel shape does not match n_cat")
            if not (np.all(np.isfinite(pi)) and np.all(pi > 0) and np.all(np.isfinite(K)) and np.all(K >= 0)
                    and np.allclose(K.sum(axis=1), 1.0, rtol=0, atol=1e-9)):
                raise ValueError("prior must be finite and positive; kernel rows must be finite, non-negative and sum to 1")
            self._logprior = np.log(pi / pi.sum())
            self._loglik_full = [np.log(p * np.eye(self.n_cat) + (1.0 - p) * K) for p in self.p_bins]

    # ---- occupancy -------------------------------------------------------------------------------------------
    def _inside(self, cells) -> np.ndarray:
        a = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
        n = self.grid.n
        return a[(a[:, 0] >= 0) & (a[:, 0] < n) & (a[:, 1] >= 0) & (a[:, 1] < n)]

    def integrate(self, obs: Observation) -> None:
        f = self._inside(obs.free)
        o = self._inside(obs.occupied)
        np.add.at(self._miss, (f[:, 0], f[:, 1]), 1)
        np.add.at(self._hit, (o[:, 0], o[:, 1]), 1)
        t = np.concatenate([f, o])
        self.occ[t[:, 0], t[:, 1]] = np.where(self._hit[t[:, 0], t[:, 1]] > self._miss[t[:, 0], t[:, 1]], OCC, FREE).astype(np.int8)
        for det in obs.detections:
            self._detect(det)

    def mark_occupied(self, rc) -> None:
        """Executor bump: the agent could not stand there; permanent whatever the scans say."""
        if self.grid.inside(rc):
            r, c = int(rc[0]), int(rc[1])
            self._hit[r, c] += 1 << 20
            self.occ[r, c] = OCC

    # ---- objects ---------------------------------------------------------------------------------------------
    def _detect(self, det) -> None:
        rc = (int(det.rc[0]), int(det.rc[1]))
        if not self.grid.inside(rc):
            return  # ponytail: object outside the fixed grid is dropped entirely (no track, no evidence)
        oid = self._id_of.get(det.key)
        if oid is None:
            oid = len(self._tracks)
            self._id_of[det.key] = oid
            self._tracks.append(Track(oid, rc, self._logprior.copy()))
        t = self._tracks[oid]
        t.rc = rc
        b, lab = int(det.dist_bin), int(det.label)
        if b in t.bins_seen:
            return  # same (instance, bin) -> same label by construction: position update only
        if self._loglik_full is not None:
            like = self._loglik_full[b][:, lab]
        else:
            off, logp = self._loglik[b]
            like = off.copy()
            like[lab] = logp
        lp = t.logpost + like
        m = lp.max()
        t.logpost = lp - (m + np.log(np.exp(lp - m).sum()))
        t.bins_seen.add(b)
        t.obs.append((b, lab))

    @property
    def n_objects(self) -> int:
        return len(self._tracks)

    def posterior(self, oid: int) -> np.ndarray:
        lp = self._tracks[oid].logpost
        p = np.exp(lp - lp.max())
        return p / p.sum()

    def label(self, oid: int) -> tuple[int, float]:
        p = self.posterior(oid)
        i = int(np.argmax(p))  # ties -> lowest class index
        return i, float(p[i])

    def near_seen(self, oid: int) -> bool:
        """A bin-0 look has been fused: no later look can add near-range evidence (one draw per instance and bin)."""
        return 0 in self._tracks[oid].bins_seen

    def eligible_revisit(self, oid: int) -> bool:
        return self.label(oid)[1] < self.tau and 0 not in self._tracks[oid].bins_seen and oid not in self.tried

    def objects_summary(self) -> list[tuple[int, int, float, tuple[int, int]]]:
        return [(t.id, *self.label(t.id), t.rc) for t in self._tracks]
