"""Vendor a few ProcTHOR-10K houses (Apache-2.0) into data/procthor/ at a pinned revision.

Tuning houses come from the train split: houses ordered by sha256("am-sp-l-tune-v1:" + index), keeping the first
N that have >= 3 rooms, a wet object (Sink, Toilet, CounterTop) and a sleep object (Bed, Dresser).
Usage: python scripts/vendor_procthor.py [--n 8]
"""
import argparse
import gzip
import hashlib
import json
import pathlib
import urllib.request

REV = "439193522244720b86d8c81cde2e51e3a4d150cf"
SHA = {"train": "ee3c4aa14b4d8f0895fecfb5fdaca59395427ca1018b2f9aeeedbc61e5824587",
       "val": "d808540514e26b6726cd2790490e669b572eeb94febb5188a2f403591dd21721"}
URL = "https://media.githubusercontent.com/media/allenai/procthor-10k/{rev}/{split}.jsonl.gz"
OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "procthor"
WET, SLEEP = {"Sink", "Toilet", "CounterTop"}, {"Bed", "Dresser"}


def fetch(split: str, cache: pathlib.Path) -> list[dict]:
    f = cache / f"{split}.jsonl.gz"
    if not f.exists():
        urllib.request.urlretrieve(URL.format(rev=REV, split=split), f)
    assert hashlib.sha256(f.read_bytes()).hexdigest() == SHA[split], f"sha256 mismatch for {f}"
    return [json.loads(line) for line in gzip.open(f)]


def passes(house: dict) -> bool:
    types = {o["id"].split("|")[0] for o in house["objects"]}
    return len(house["rooms"]) >= 3 and bool(types & WET) and bool(types & SLEEP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cache", default=str(OUT / ".cache"))
    a = ap.parse_args()
    cache = pathlib.Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    houses = fetch("train", cache)
    order = sorted(range(len(houses)), key=lambda i: hashlib.sha256(f"am-sp-l-tune-v1:{i}".encode()).hexdigest())
    picked = [i for i in order if passes(houses[i])][: a.n]
    with open(OUT / "tune.jsonl", "w") as f:
        for i in picked:
            f.write(json.dumps({"scene": f"train-{i:05d}", "house": houses[i]}, sort_keys=True) + "\n")
    print("vendored", [f"train-{i:05d}" for i in picked])


if __name__ == "__main__":
    main()
