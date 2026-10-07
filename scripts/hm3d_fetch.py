"""Fetch HM3D v0.2 data on the AWS instance only (never on the Mac, never into the repo).

Credentials (HM3D_TOKEN_ID, HM3D_TOKEN_SECRET) come from ~/amap/.secrets.env (mode 600, written by deploy.sh from
stdin, never exported) or the environment, and are never printed. The Matterport endpoint redirects to object storage; Authorization is sent only to the first host.
If the storage honours HTTP Range, a remote tar is read like a seekable file, so a few scenes can be pulled out of
the 27 GB train archive without downloading it.

  python scripts/hm3d_fetch.py probe                                  # Range support + throughput
  python scripts/hm3d_fetch.py list semantic-annots --split train     # scene folders in an archive
  python scripts/hm3d_fetch.py get habitat --split train --scenes 00006-HkseAnWCgqk --dest DIR
  python scripts/hm3d_fetch.py all semantic-annots --split val --dest DIR
"""
from __future__ import annotations

import argparse
import base64
import http.client
import io
import os
import pathlib
import re
import sys
import tarfile
import time
import urllib.parse
import urllib.request

BASE = "https://api.matterport.com/resources/habitat/"
NAMES = {"habitat": "hm3d-{split}-habitat-v0.2.tar", "semantic-annots": "hm3d-{split}-semantic-annots-v0.2.tar",
         "semantic-configs": "hm3d-{split}-semantic-configs-v0.2.tar", "configs": "hm3d-{split}-configs.tar"}


def secret(name: str) -> str:
    """Environment first (local runs); on the AWS instance the value lives only in ~/amap/.secrets.env (mode 600),
    which remote_run.sh validates but never exports, so no shell or log in the job ever holds it."""
    if os.environ.get(name):
        return os.environ[name]
    f = pathlib.Path(os.environ.get("AMAP_SECRETS", pathlib.Path.home() / "amap" / ".secrets.env"))
    if f.exists():
        for line in f.read_text().splitlines():
            k, _, v = line.partition("=")
            if k == name and v:
                return v
    raise KeyError(f"{name} not set")


def resolve(component: str, split: str) -> tuple[str, int, bool]:
    """Follow the authenticated redirect once; return (final URL, size, range supported)."""
    url = BASE + NAMES[component].format(split=split)
    tok = base64.b64encode(f"{secret('HM3D_TOKEN_ID')}:{secret('HM3D_TOKEN_SECRET')}".encode()).decode()
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    req.add_unredirected_header("Authorization", "Basic " + tok)  # never forwarded to the storage host
    with urllib.request.urlopen(req, timeout=60) as r:
        final, status = r.geturl(), r.status
        cr = r.headers.get("Content-Range", "")
        size = int(cr.rsplit("/", 1)[1]) if "/" in cr else int(r.headers.get("Content-Length", 0))
    return final, size, status == 206


class RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP Range requests on one keep-alive connection."""

    def __init__(self, url: str, size: int, chunk=1 << 20):
        self.u = urllib.parse.urlsplit(url)
        self.size, self.pos, self.chunk = size, 0, chunk
        self.buf_start, self.buf = 0, b""
        self.conn = None
        self.requests = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _get(self, start, end):
        path = self.u.path + ("?" + self.u.query if self.u.query else "")
        for attempt in range(5):
            try:
                if self.conn is None:
                    self.conn = http.client.HTTPSConnection(self.u.netloc, timeout=60)
                self.conn.request("GET", path, headers={"Range": f"bytes={start}-{end}"})
                r = self.conn.getresponse()
                data = r.read()
                if r.status != 206:
                    raise IOError(f"HTTP {r.status} for range request")
                self.requests += 1
                return data
            except (http.client.HTTPException, OSError):
                self.conn = None
                time.sleep(2 ** attempt)
        raise IOError("range request failed 5 times")

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        if not (self.buf_start <= self.pos and self.pos + n <= self.buf_start + len(self.buf)):
            want = max(n, self.chunk if n < self.chunk else n)
            self.buf_start, self.buf = self.pos, self._get(self.pos, min(self.size, self.pos + want) - 1)
        o = self.pos - self.buf_start
        b[:n] = self.buf[o:o + n]
        self.pos += n
        return n


def open_remote_tar(component, split):
    url, size, ok = resolve(component, split)
    if not ok:
        sys.exit(f"{component}/{split}: storage does not honour Range; use `all` (full download) instead")
    rf = RangeFile(url, size)
    return tarfile.open(fileobj=io.BufferedReader(rf, buffer_size=1 << 16), mode="r:"), rf, size


SCENE = re.compile(r"^\d{5}-[A-Za-z0-9]{11}$")  # e.g. 00800-TEEsavR23oF


def scene_of(name: str):
    return next((part for part in name.split("/") if SCENE.match(part)), None)


def scene_dirs(members):
    return sorted({scene_of(m.name) for m in members} - {None})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "list", "get", "all"])
    ap.add_argument("component", nargs="?", default="habitat", choices=list(NAMES))
    ap.add_argument("--split", default="val")
    ap.add_argument("--scenes", nargs="*", default=[])
    ap.add_argument("--dest", default="hm3d")
    a = ap.parse_args()
    if a.cmd == "probe":
        url, size, ok = resolve("habitat", "train")
        print(f"train habitat: {size / 1e9:.1f} GB, range {'yes' if ok else 'no'}, host {urllib.parse.urlsplit(url).netloc}")
        t0 = time.time()
        rf = RangeFile(url, size, chunk=64 << 20)
        got = 0
        while got < (256 << 20):
            rf.seek(got)
            got += len(rf.read(64 << 20))
        print(f"throughput {got / (time.time() - t0) / 1e6:.0f} MB/s over {got >> 20} MB")
        return
    if a.cmd == "all":
        url, size, _ = resolve(a.component, a.split)
        t0 = time.time()
        dest = pathlib.Path(a.dest); dest.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r, tarfile.open(fileobj=r, mode="r|") as tar:
            tar.extractall(dest)
        print(f"{a.component}/{a.split}: {size / 1e9:.2f} GB in {time.time() - t0:.0f} s -> {dest}")
        return
    tar, rf, size = open_remote_tar(a.component, a.split)
    t0 = time.time()
    members = tar.getmembers()
    print(f"{a.component}/{a.split}: {len(members)} members, index read with {rf.requests} range requests in {time.time() - t0:.0f} s")
    if a.cmd == "list":
        for s in scene_dirs(members):
            print(s)
        return
    want = [m for m in members if scene_of(m.name) in set(a.scenes)]
    missing = set(a.scenes) - {scene_of(m.name) for m in want}
    if missing:
        sys.exit(f"not in archive: {sorted(missing)}")
    tar.extractall(a.dest, members=want)
    print(f"extracted {len(want)} members for {len(a.scenes)} scenes -> {a.dest}")


if __name__ == "__main__":
    main()
