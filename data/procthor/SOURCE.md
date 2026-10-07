# ProcTHOR-10K houses (vendored)

- Source: https://github.com/allenai/procthor-10k at revision 439193522244720b86d8c81cde2e51e3a4d150cf (Git LFS objects `train.jsonl.gz`, sha256 ee3c4aa1…; `val.jsonl.gz`, sha256 d8085405…).
- License: Apache-2.0 (`LICENSE` in this folder, copied unchanged from that revision).
- `tune.jsonl`: houses picked from the train split by `scripts/vendor_procthor.py` (hash-ordered, filtered); each line is `{"scene": "train-<index>", "house": <unmodified house JSON>}`.
