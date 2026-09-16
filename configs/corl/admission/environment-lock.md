# T710 environment lock — CoRL / Janus-Pro-1B admission

Recorded: 2026-09-16T13:15Z, remote-gpu-agent.

## Dev machine (this session)

This session's shell is not GPU-attached. All research on the public code
and paper was done here read-only against a shallow scratch clone at
`/tmp/corl_audit` (git HEAD `0c92629f9b307a32bb286ae3562809e941d1bb0b`,
**not committed** to this branch). PDF extraction of arXiv:2505.17534 v3 was
done with `pymupdf`, installed locally via `pip install --user pymupdf` in
this dev session only (not part of any training environment; used only to
read the paper text).

## GPU execution target: H20-FoldUMM

- `task_flag`: `gpu_model_train_split_llm218029E40DDAA49`
- `instance_id`: `8b1d81c89f83d4d4019f93a9170a251d`
- 8x NVIDIA H20 96GB (confirmed via `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu`)
- GPU0: `0 MiB used / 97871 MiB total, 0% util` — free, reserved for T710.
- GPU1-7: `325 MiB used, 100% util` each — this is the documented benign
  `train2.py` placeholder daemon, not real work; not touched.
- Access method: `taiji_client exec` wrapped in a fake PTY
  (`script -qec "taiji_client exec -scfg '<SCFG>' '<task_flag>' '<instance_id>' bash -c '<cmd>'" /dev/null 2>&1 | tr -d '\r'`),
  because `taiji_client exec` requires a PTY.
- Long-running commands (venv build, downloads, GPU smoke) go through
  `cjob.sh` (`/apdcephfs_cq9/share_1447896/yihangli/workspace/taiji_gpu_start/scripts/cjob.sh`)
  since a plain `exec` session idle-times-out around 30s and does not
  survive disconnects; `nohup`/`&` alone do not survive either.

### SCFG discrepancy (documented, not silently substituted)

The operator briefing for this task specified
`SCFG=/apdcephfs_cq9/share_1447896/yihangli/workspace/taiji_gpu_start/configs/config_h20_llm2.json`.
This project's own `taiji_gpu_start/SERVER_STATUS.md` and
`taiji_gpu_start/scripts/connect_foldumm.sh` both specify
`config_h20_foldumm.json` as the correct config for H20-FoldUMM specifically
(matching the exact `task_flag`/`instance_id` given for this task). I
compared both JSON files: they share an identical `Token` and
`business_flag` and differ only in the `envs` block (DATA_ROOT/CODE_ROOT/task
description strings), so `exec` behavior is very likely unaffected either
way. I used `config_h20_foldumm.json` (verified working: successfully
returned hostname/df/nvidia-smi output below) rather than the briefing's
`config_h20_llm2.json`, on the basis that the project's own documented
per-container config file is the more reliable source of truth than a
possibly-stale operator note. Flagging this explicitly rather than silently
deviating.

## Local SSD verification (contract requires `filesystem_class == local`)

Ran inside the H20-FoldUMM container:

```
$ df -hT /dockerdata /apdcephfs_cq7 /apdcephfs_cq9
Filesystem                 Type     Size  Used Avail Use% Mounted on
/dev/mapper/gpu-gpu_volume xfs      9.0T  159G  8.9T   2% /dockerdata
```

`/dockerdata` is XFS on `/dev/mapper/gpu-gpu_volume`, backed by real local
NVMe block devices inside the container (confirmed via `lsblk`); it is not a
ceph/dop-fuse mount. `/apdcephfs_cq7` and `/apdcephfs_cq9` are network
(ceph-fuse) mounts and are NOT used for the pinned assets, venv, or smoke
working directory — only for this repo's git worktree itself (which lives on
`/apdcephfs_cq9` per the task's own governance, outside the storage
preflight's scope) and for durable outputs/logs via `cjob.sh`'s job root.

All pinned assets (Janus-Pro-1B weights, x2x_rft_22k micro-split,
all-mpnet-base-v2, the dedicated Python venv) will be placed under
`/dockerdata/t710-corl/`, matching the convention used by every prior
admission task in this container.

## Python / build toolchain observed in-container

- `python3 --version` → `Python 3.10.12`
- `which conda` → empty (no conda binary available; the official
  `Install.md`'s `conda create -n corl` instructions are adapted to a plain
  `python3 -m venv` at `/root/venvs/corl` instead — kept separate from a
  pre-existing, unrelated `/root/venvs/janus_pro` venv found in this
  container, which has a broken `huggingface-hub` pin
  (`huggingface-hub==1.29.0` present, but `transformers` requires
  `>=0.34.0,<1.0`) and appears to belong to a different task; not reused, not
  repaired, to avoid collision risk with concurrently running tasks (e.g.
  T720 on this same container).
- Exact resolved dependency versions for the fresh `/root/venvs/corl` venv
  will be appended here (or in a follow-up note) after the venv is built,
  ahead of the GPU smoke run.

## Concurrency note

T720 (`agent/T720-janus-pro-r1-stack-smoke`) is a sibling admission task that
may run concurrently on this same H20-FoldUMM container on a different GPU
index. T710 exclusively uses GPU index 0. No shared mutable state (venv
path, asset path, job names) overlaps with T720's expected `/dockerdata/`
namespace, since T710 uses the `t710-corl` prefix throughout.
