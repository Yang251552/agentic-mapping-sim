"""Pictures for the anchor output: a run's map.png and the side-by-side demo GIF of two recorded runs.

  python -m amap.viz png RUN_DIR       # re-render RUN_DIR/map.png from map.npz + trajectory.json
  python -m amap.viz gif --left D1 --right D2 --left-b1 D3 --right-b1 D4 --out figures/demo.gif

Pillow + numpy only. Image orientation: row r grows downward, column c to the right (+x right, +y down).
frames.npz: `occ` int8 (F, H, W) crops of the n x n occupancy grid at the final known bounding box + margin, and
`meta`, a JSON string (uint8 bytes) with n, cell, start, offset (crop origin r0, c0), steps_used, end_reason and one
entry per frame. Frame F-1 is an extra "final" frame (the map after the last decision, no candidates), so the GIF
ends on the finished map and the full Q_T curve; the other frames are the recorded decisions.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .contracts import CATEGORIES, FREE, OCC, UNKNOWN

PX = 4            # map pixels per grid cell
MARGIN_M = 2.0    # margin around the known area
TAU = 0.7         # objects below this confidence are drawn hollow

BLACK, WHITE = (0, 0, 0), (255, 255, 255)
_LUT = np.zeros((3, 3), dtype=np.uint8)     # index occ + 1 -> RGB
_LUT[UNKNOWN + 1], _LUT[FREE + 1], _LUT[OCC + 1] = (0, 0, 0), (205, 205, 205), (70, 70, 70)
LABEL_COLOR = {"sink": (31, 119, 180), "toilet": (44, 160, 44), "counter": (140, 86, 75), "shower": (23, 190, 207),
               "bathtub": (148, 103, 189), "bed": (214, 39, 40), "chest_of_drawers": (227, 119, 194)}
OTHER = (165, 165, 165)
TRAJ = (0, 70, 220)
FRONTIER, REVISIT = (255, 225, 0), (255, 150, 0)
CHOSEN = (235, 0, 0)
LLM_C, B1_C = (0, 90, 200), (120, 120, 120)

_fonts: dict = {}


def _font(size):
    if size not in _fonts:
        try:
            _fonts[size] = ImageFont.load_default(size=size)
        except TypeError:  # Pillow < 10.1
            _fonts[size] = ImageFont.load_default()
    return _fonts[size]


def _text(d, xy, s, size, fill=BLACK, halo=None, anchor="la"):
    kw = {"stroke_width": 2, "stroke_fill": halo} if halo else {}
    try:
        d.text(xy, s, font=_font(size), fill=fill, anchor=anchor, **kw)
    except (TypeError, ValueError):  # bitmap fallback font: no anchor / stroke
        d.text(xy, s, font=_font(size), fill=fill)


def _tw(d, s, size) -> float:
    return d.textlength(s, font=_font(size))


# ---- map layers -------------------------------------------------------------------------------------------------
def _box(occ, margin):
    """(r0, r1, c0, c1), end exclusive: bounding box of the known cells plus margin cells, clipped to the grid."""
    known = occ != UNKNOWN
    rows, cols = np.flatnonzero(known.any(1)), np.flatnonzero(known.any(0))
    n0, n1 = occ.shape
    if len(rows) == 0:
        return 0, n0, 0, n1
    return (max(rows[0] - margin, 0), min(rows[-1] + 1 + margin, n0), max(cols[0] - margin, 0), min(cols[-1] + 1 + margin, n1))


def _margin(cell) -> int:
    return int(round(MARGIN_M / cell))


def _base(occ, px) -> Image.Image:
    h, w = occ.shape
    small = Image.fromarray(_LUT[occ.astype(np.intp) + 1])
    return small.resize((max(1, round(w * px)), max(1, round(h * px))), Image.NEAREST)


def _xy(rc, off, px):
    return ((rc[1] - off[1] + 0.5) * px, (rc[0] - off[0] + 0.5) * px)


def _layers(d, off, px, objs, traj, start):
    """Trajectory polyline, objects (filled when conf >= TAU, hollow below), start marker. objs: (oid, cat, conf, rc)."""
    if len(traj) >= 2:
        d.line([_xy(p, off, px) for p in traj], fill=TRAJ, width=1 if px < 6 else 2)
    h = 4.0
    for _, cat, conf, rc in objs:
        x, y = _xy(rc, off, px)
        col = LABEL_COLOR.get(CATEGORIES[cat], OTHER)
        if conf >= TAU:
            d.rectangle([x - h, y - h, x + h, y + h], fill=col, outline=BLACK)
        else:
            d.rectangle([x - h - 1, y - h - 1, x + h + 1, y + h + 1], outline=BLACK, width=4)
            d.rectangle([x - h, y - h, x + h, y + h], outline=col, width=2)
    x, y = _xy(start, off, px)
    d.polygon([(x, y - 6), (x + 6, y + 5), (x - 6, y + 5)], fill=(40, 200, 60), outline=BLACK)


def render_map(occ, objs, traj, start, cell, px=PX) -> Image.Image:
    """Known area + 2 m margin of a full n x n occ grid. traj: [[step, r, c, heading], ...]."""
    r0, r1, c0, c1 = _box(occ, _margin(cell))
    img = _base(occ[r0:r1, c0:c1], px)
    _layers(ImageDraw.Draw(img), (r0, c0), px, objs, [(t[1], t[2]) for t in traj], start)
    return img


def save_map_png(path, res):
    amap = res["amap"]
    render_map(amap.occ, amap.objects_summary(), res["trajectory"], amap.grid.start, amap.grid.cell).save(path)


# ---- frames.npz -------------------------------------------------------------------------------------------------
def save_frames(path, res):
    amap = res["amap"]
    r0, r1, c0, c1 = _box(amap.occ, _margin(amap.grid.cell))
    tr = res["trajectory"]
    end = {"step": int(res["steps_used"]), "occ": amap.occ, "rc": tr[-1][1:3] if tr else amap.grid.start, "cands": [],
           "cid": None, "nearest": None, "reason": "", "objects": amap.objects_summary(), "final": True}
    frames = list(res["frames"]) + [end]
    meta = {"n": int(amap.grid.n), "cell": float(amap.grid.cell), "start": [int(v) for v in amap.grid.start],
            "offset": [int(r0), int(c0)], "steps_used": int(res["steps_used"]), "end_reason": res.get("end_reason", ""),
            "frames": [{"step": int(f["step"]), "rc": [int(v) for v in f["rc"]], "cid": f["cid"], "nearest": f["nearest"],
                        "reason": f["reason"], "final": bool(f.get("final")),
                        "cands": [[c.cid, c.kind, [int(v) for v in c.target], len(c.cells), [int(o) for o in c.objects]]
                                  for c in f["cands"]],
                        "objects": [[int(o), int(k), float(p), [int(v) for v in rc]] for o, k, p, rc in f["objects"]]}
                       for f in frames]}
    np.savez_compressed(path, occ=np.stack([f["occ"][r0:r1, c0:c1] for f in frames]).astype(np.int8),
                        meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8))


def load_frames(path):
    """-> (frames, info). info: n, cell, start, offset (r0, c0 of the crop), steps_used, end_reason.
    frames: list of dicts step, rc, cands [[cid, kind, [r, c], n_cells, [object ids]]], cid, nearest, reason, objects
    [[oid, cat, conf, [r, c]]], final (True on the last, post-decision frame), occ (int8 crop)."""
    with np.load(path) as z:
        stack, info = z["occ"], json.loads(z["meta"].tobytes().decode())
    frames = info.pop("frames")
    for f, o in zip(frames, stack):
        f["occ"] = o
    return frames, info


# ---- GIF --------------------------------------------------------------------------------------------------------
HEAD, REASON_H, CHART_H, GAP, PAD, LEG_H = 26, 40, 104, 16, 10, 24
MIN_PANEL_W = 460


def _arrow(d, p0, p1, color, width, dashed):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy)
    if L < 1:
        return
    ux, uy = dx / L, dy / L
    tip = (p1[0] - ux * 7, p1[1] - uy * 7)               # stop short of the candidate marker
    segs = [(0.0, max(L - 7, 0.0))]
    if dashed:
        segs = [(a, min(a + 6, L - 7)) for a in np.arange(0, L - 7, 10.0)]
    head = [tip, (tip[0] - ux * 7 - uy * 4, tip[1] - uy * 7 + ux * 4), (tip[0] - ux * 7 + uy * 4, tip[1] - uy * 7 - ux * 4)]
    for col, w in ((BLACK, width + 2), (color, width)):
        for a, b in segs:
            d.line([(p0[0] + ux * a, p0[1] + uy * a), (p0[0] + ux * b, p0[1] + uy * b)], fill=col, width=w)
        d.polygon(head, fill=col)


def _marker(d, c, p, chosen):
    x, y = p
    fill = FRONTIER if c[1] == "frontier" else REVISIT
    for r, f, o in (((8, CHOSEN, BLACK), (5, fill, BLACK)) if chosen else ((5, fill, BLACK),)):
        if c[1] == "frontier":
            d.ellipse([x - r, y - r, x + r, y + r], fill=f, outline=o)
        else:
            d.polygon([(x, y - r - 1), (x + r + 1, y), (x, y + r + 1), (x - r - 1, y)], fill=f, outline=o)


def _wrap2(d, text, size, width):
    """Greedy word wrap to two lines; the second is ellipsized when the text does not fit."""
    words, l1, i = text.split(), "", 0
    while i < len(words) and _tw(d, (l1 + " " + words[i]).strip(), size) <= width:
        l1 = (l1 + " " + words[i]).strip()
        i += 1
    l2 = " ".join(words[i:])
    if _tw(d, l2, size) > width:
        while l2 and _tw(d, l2 + "...", size) > width:
            l2 = l2[:-1]
        l2 = l2.rstrip() + "..."
    return l1, l2


def _series(curve, ended, step):
    pts = [(c["step"], c["q_t"]) for c in curve if c["step"] <= step]
    if pts and ended < step:      # the run stopped: its map, hence Q_T, no longer changes
        pts.append((step, pts[-1][1]))
    return pts


def _chart(w, run, key, b1key, step, xmax):
    img = Image.new("RGB", (w, CHART_H), WHITE)
    d = ImageDraw.Draw(img)
    L, R, T, B = 30, 8, 24, 16
    px = lambda s: L + s / xmax * (w - L - R)
    py = lambda q: T + (1 - q) * (CHART_H - T - B)
    for q in (0.0, 0.5, 1.0):
        d.line([(L, py(q)), (w - R, py(q))], fill=(220, 220, 220))
        _text(d, (L - 4, py(q)), f"{q:g}", 10, (90, 90, 90), anchor="rm")
    for s in range(0, int(xmax) + 1, 100):
        _text(d, (px(s), CHART_H - B + 2), str(s), 10, (90, 90, 90), anchor="ma")
    d.line([(px(step), T), (px(step), CHART_H - B)], fill=(190, 190, 190))
    x = L
    _text(d, (2, 2), "Q_T", 11, (60, 60, 60))
    for label, col, pts in (("LLM", LLM_C, _series(run[key], run["info"]["steps_used"], step)),
                            ("nearest frontier", B1_C, _series(run[b1key], run["b1_end"], step))):
        if len(pts) >= 2:
            d.line([(px(s), py(q)) for s, q in pts], fill=col, width=2)
        val = f" {pts[-1][1]:.2f}" if pts else ""
        d.line([(x + 36, 9), (x + 52, 9)], fill=col, width=2)
        _text(d, (x + 56, 2), label + val, 11, col)
        x += 62 + _tw(d, label + val, 11)
    return img


def _panel(run, k, canvas_off, px, title, budget, occ_pad):
    fr = run["frames"][min(k, len(run["frames"]) - 1)]
    occ = occ_pad[min(k, len(occ_pad) - 1)]
    off = canvas_off
    step = fr["step"]
    img = _base(occ, px)
    d = ImageDraw.Draw(img)
    traj = [(t[1], t[2]) for t in run["traj"] if t[0] <= step] or [fr["rc"]]
    objs = [(o[0], o[1], o[2], tuple(o[3])) for o in fr["objects"]]
    _layers(d, off, px, objs, traj, run["info"]["start"])
    by = {c[0]: c for c in fr["cands"]}
    me, near = _xy(fr["rc"], off, px), by.get(fr["nearest"])
    ch = by.get(fr["cid"])
    if ch and near and near[0] != ch[0]:
        _arrow(d, me, _xy(near[2], off, px), WHITE, 2, True)
    if ch:
        _arrow(d, me, _xy(ch[2], off, px), CHOSEN, 2, False)
    for c in fr["cands"]:
        p = _xy(c[2], off, px)
        _marker(d, c, p, c is ch)
        _text(d, (p[0] + 8, p[1] - 15), c[0], 10, BLACK, halo=WHITE)
    revisit = bool(ch and ch[1] == "revisit")
    if revisit:   # ring the objects that this revisit re-observes, tag the target
        for o in objs:
            if o[0] in ch[4]:
                x, y = _xy(o[3], off, px)
                d.ellipse([x - 10, y - 10, x + 10, y + 10], outline=REVISIT, width=2)
        tx, ty = _xy(ch[2], off, px)
        d.rectangle([tx + 8, ty + 4, tx + 8 + 52, ty + 18], fill=REVISIT, outline=BLACK)
        _text(d, (tx + 12, ty + 5), "REVISIT", 10, BLACK)
    me_r = 6
    d.ellipse([me[0] - me_r - 1, me[1] - me_r - 1, me[0] + me_r + 1, me[1] + me_r + 1], fill=WHITE, outline=BLACK, width=2)
    d.ellipse([me[0] - 3, me[1] - 3, me[0] + 3, me[1] + 3], fill=(0, 110, 255))
    mw, hm = img.size
    w = max(mw, MIN_PANEL_W)      # a small map is centred on black so the text lines still fit
    pan = Image.new("RGB", (w, HEAD + hm + REASON_H + CHART_H), WHITE)
    pd = ImageDraw.Draw(pan)
    pd.rectangle([0, 0, w, HEAD - 1], fill=(30, 30, 40))
    _text(pd, (6, 4), title, 16, WHITE)
    _text(pd, (6 + _tw(pd, title, 16) + 14, 7), f"step {step} / {budget}", 13, (210, 210, 210))
    if revisit:
        pd.rectangle([w - 64, 4, w - 5, HEAD - 5], fill=REVISIT)
        _text(pd, (w - 60, 6), "REVISIT", 12, BLACK)
    pd.rectangle([0, HEAD, w, HEAD + hm - 1], fill=BLACK)
    pan.paste(img, ((w - mw) // 2, HEAD))
    reason = f"end of episode ({run['info']['end_reason']}), {run['info']['steps_used']} steps" if fr["final"] else fr["reason"]
    for i, line in enumerate(_wrap2(pd, reason, 12, w - 8)):
        _text(pd, (4, HEAD + hm + 3 + i * 16), line, 12, (40, 40, 40))
    pan.paste(_chart(w, run, "curve", "b1", step, max(budget, 1)), (0, HEAD + hm + REASON_H))
    return pan


def _legend(width):
    items = [("sq", c, n) for n, c in LABEL_COLOR.items()] + [("sq", OTHER, "other"), ("hol", (60, 60, 60), "hollow = conf < 0.7"),
             ("circ", FRONTIER, "frontier"), ("dia", REVISIT, "revisit"), ("sol", CHOSEN, "chosen"), ("das", WHITE, "nearest frontier"),
             ("tri", (40, 200, 60), "start")]
    img = Image.new("RGB", (width, LEG_H * 4), WHITE)
    d = ImageDraw.Draw(img)
    x, y = 6, 4
    for kind, col, name in items:
        tw = _tw(d, name, 11)
        if x + 22 + tw > width - 6:
            x, y = 6, y + LEG_H - 4
        cx, cy = x + 7, y + 8
        if kind == "sq":
            d.rectangle([cx - 5, cy - 5, cx + 5, cy + 5], fill=col, outline=BLACK)
        elif kind == "hol":
            d.rectangle([cx - 5, cy - 5, cx + 5, cy + 5], outline=col, width=2)
        elif kind == "circ":
            d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=col, outline=BLACK)
        elif kind == "dia":
            d.polygon([(cx, cy - 6), (cx + 6, cy), (cx, cy + 6), (cx - 6, cy)], fill=col, outline=BLACK)
        elif kind == "tri":
            d.polygon([(cx, cy - 6), (cx + 6, cy + 5), (cx - 6, cy + 5)], fill=col, outline=BLACK)
        else:
            _arrow(d, (cx - 9, cy), (cx + 16, cy), col, 2, kind == "das")
        _text(d, (x + 22, y + 1), name, 11, (40, 40, 40))
        x += 22 + tw + 14
    return img.crop((0, 0, width, y + LEG_H - 2))


def _load_run(d, b1):
    d, b1 = pathlib.Path(d), pathlib.Path(b1)
    frames, info = load_frames(d / "frames.npz")
    rj = lambda p: json.loads(p.read_text())
    return {"frames": frames, "info": info, "traj": rj(d / "trajectory.json"), "curve": rj(d / "curve.json"),
            "b1": rj(b1 / "curve.json"), "b1_end": rj(b1 / "metrics.json")["steps_used"]}


def _pad(runs):
    """Common crop (union of the runs' crops) and each run's occ stack padded to it with UNKNOWN."""
    boxes = [(r["info"]["offset"][0], r["info"]["offset"][1], *r["frames"][0]["occ"].shape) for r in runs]
    R0, C0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    R1, C1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    out = []
    for r, (r0, c0, h, w) in zip(runs, boxes):
        st = np.full((len(r["frames"]), R1 - R0, C1 - C0), UNKNOWN, dtype=np.int8)
        for i, f in enumerate(r["frames"]):
            st[i, r0 - R0:r0 - R0 + h, c0 - C0:c0 - C0 + w] = f["occ"]
        out.append(st)
    return (R0, C0), out


def _write_gif(frames, out, ms):
    step = max(1, len(frames) // 12)    # one shared palette from a sample of frames: no colour flicker, small deltas
    pal = Image.fromarray(np.vstack([np.asarray(f) for f in frames[::step]])).quantize(256, dither=0)
    pf = [f.quantize(palette=pal, dither=0) for f in frames]
    pf[0].save(out, save_all=True, append_images=pf[1:], duration=[ms] * (len(pf) - 1) + [max(ms, 3000)], loop=0, disposal=1)


def make_gif(out_path, left_dir, right_dir, left_b1_dir, right_b1_dir, titles=("T_wet", "T_sleep"), ms_per_frame=500,
             budget=None, max_mb=8.0):
    """Two recorded LLM runs side by side with their nearest-frontier (B1) Q_T curves underneath.
    budget: shown as "step k / budget" and the chart's x range; default = the largest steps_used of the four runs.
    The map scale starts at 4 px per cell (up to 8 for small maps, so a panel is >= 460 px wide) and drops by 1 until the
    file is <= max_mb."""
    runs = [_load_run(left_dir, left_b1_dir), _load_run(right_dir, right_b1_dir)]
    budget = int(budget or max(max(r["info"]["steps_used"], r["b1_end"]) for r in runs))
    off, pads = _pad(runs)
    hc, wc = pads[0].shape[1:]
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    nf = max(len(r["frames"]) for r in runs)
    px0 = min(max(PX, math.ceil(MIN_PANEL_W / wc)), 8)    # small houses: scale up so the text lines stay readable
    for px in range(px0, 1, -1):
        frames = []
        for k in range(nf):
            a, b = (_panel(r, k, off, px, t, budget, pd) for r, t, pd in zip(runs, titles, pads))
            leg = _legend(a.width + b.width + GAP)
            im = Image.new("RGB", (a.width + b.width + GAP + 2 * PAD, a.height + leg.height + 2 * PAD), WHITE)
            im.paste(a, (PAD, PAD))
            im.paste(b, (PAD + a.width + GAP, PAD))
            im.paste(leg, (PAD, PAD + a.height + 4))
            frames.append(im)
        _write_gif(frames, out, ms_per_frame)
        if out.stat().st_size <= max_mb * 1e6:
            break
    return {"frames": nf, "size": frames[0].size, "px": px, "mb": out.stat().st_size / 1e6}


# ---- CLI --------------------------------------------------------------------------------------------------------
def png_from_run(run_dir, cell=0.2):
    d = pathlib.Path(run_dir)
    with np.load(d / "map.npz") as z:
        occ, rc, post = z["occ"], z["obj_rc"], z["obj_posterior"]
    objs = [(i, int(p.argmax()), float(p.max()), (int(r[0]), int(r[1]))) for i, (r, p) in enumerate(zip(rc, post))]
    n = occ.shape[0]
    render_map(occ, objs, json.loads((d / "trajectory.json").read_text()), (n // 2, n // 2), cell).save(d / "map.png")
    return d / "map.png"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m amap.viz")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gif")
    for k in ("left", "right", "left-b1", "right-b1", "out"):
        g.add_argument("--" + k, required=True)
    g.add_argument("--titles", nargs=2, default=["T_wet", "T_sleep"])
    g.add_argument("--ms", type=int, default=500)
    g.add_argument("--budget", type=int)
    p = sub.add_parser("png")
    p.add_argument("run_dir")
    p.add_argument("--cell", type=float, default=0.2, help="grid cell size in metres (sets the 2 m margin)")
    a = ap.parse_args(argv)
    if a.cmd == "png":
        print(png_from_run(a.run_dir, a.cell))
    else:
        print(make_gif(a.out, a.left, a.right, a.left_b1, a.right_b1, tuple(a.titles), a.ms, a.budget))
    return 0


if __name__ == "__main__":
    sys.exit(main())
