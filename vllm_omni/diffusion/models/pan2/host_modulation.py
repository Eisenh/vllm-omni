# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Host-resident adaLN modulation for PAN 2.

Each PAN 2 joint block owns an FP32 modulation linear ``norm1.linear`` (4096 -> 6 * 4096). Its input,
``silu(temb)``, depends on the denoising timestep only, so for a given timestep the modulation of every block is a
constant ``[num_blocks, 6 * dim]`` table (4.3 MB for the released checkpoint). The 44 linears together are 4.43B
FP32 parameters (17.7 GB) and, as replicated linears, sit in full on every tensor-parallel rank.

With host modulation the linears are never built on the accelerator. Their weights are memory-mapped read-only
from the checkpoint's safetensors files (so all ranks on a node share one page-cache copy), the per-timestep table
is computed on the CPU in FP32 and cached in pinned host memory, and each forward uploads the table of its timestep.

``model_config.pan2_host_modulation`` is ``"auto"`` by default: host modulation turns on when the replicated
modulation would take at least a quarter of the accelerator's memory (GPUs of up to 70 GB for the released
checkpoint) and no layerwise offload streams the blocks. Every other configuration, including all profiles of
the PAN 2 recipes for 48 GB and larger GPUs, keeps the original path. ``true`` or ``false`` forces it.

This differs from ``ExactProjectionCache`` (which keeps weights where they are) and from MiniMax-H3's AdaLN offload
(which builds CPU linears, so each tensor-parallel worker process would hold its own 17.7 GB host copy).
"""

from __future__ import annotations

import json
import math
import os
import struct
import threading
import warnings
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import regex as re
import torch
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.transformers_utils.repo_utils import hf_api

logger = init_logger(__name__)

HOST_MODULATION_KEY = "pan2_host_modulation"
HOST_MODULATION_CACHE_MB_KEY = "pan2_host_modulation_cache_mb"
HOST_MODULATION_THREADS_KEY = "pan2_host_modulation_threads"
# "auto" enables host modulation when the replicated modulation is at least this share of accelerator memory.
AUTO_MEMORY_FRACTION = 0.25

_MODULATION_RE = re.compile(r"^transformer_blocks\.(\d+)\.norm1\.linear\.(weight|bias)$")
_SAFETENSORS_DTYPES = {"F32": np.float32}


def is_host_modulation_weight(name: str) -> bool:
    return _MODULATION_RE.match(name) is not None


def modulation_bytes(num_blocks: int, dim: int) -> int:
    """FP32 bytes of all ``norm1.linear`` weights and biases, as held by each tensor-parallel rank."""
    return num_blocks * 6 * dim * (dim + 1) * 4


def host_modulation_enabled(
    od_config: object, *, num_blocks: int, dim: int, device_memory: int | None, streams_blocks: bool
) -> bool:
    """Resolve ``model_config.pan2_host_modulation`` (``"auto"``, ``True`` or ``False``).

    ``device_memory`` is the accelerator's total memory in bytes, or ``None`` when unknown. ``streams_blocks`` is
    whether a layerwise offload policy streams the blocks. ``"auto"`` keeps the original path when the memory is
    unknown or the blocks are streamed.
    """
    model_config = getattr(od_config, "model_config", None) or {}
    value = model_config.get(HOST_MODULATION_KEY, "auto")
    if isinstance(value, bool):
        return value
    if value != "auto":
        raise ValueError(f'model_config.{HOST_MODULATION_KEY} must be true, false or "auto", got {value!r}')
    if not device_memory or streams_blocks:
        return False
    needed = modulation_bytes(num_blocks, dim)
    enabled = needed >= AUTO_MEMORY_FRACTION * device_memory
    if enabled:
        logger.info(
            "PAN2 host modulation enabled automatically: the replicated modulation (%.1f GiB) would take %.0f%% of "
            "the accelerator's %.1f GiB. Set model_config.%s=false to keep it on the accelerator.",
            needed / 2**30,
            100 * needed / device_memory,
            device_memory / 2**30,
            HOST_MODULATION_KEY,
        )
    return enabled


def resolve_transformer_dir(model: str, revision: str | None = None) -> Path:
    """Local directory that holds the transformer safetensors of ``model`` (a path or a Hub repo id)."""
    local = Path(model) / "transformer"
    if local.is_dir():
        return local
    kwargs = {"revision": revision, "allow_patterns": ["transformer/*"]}
    try:
        root = hf_api().snapshot_download(model, local_files_only=True, **kwargs)
    except FileNotFoundError:  # not cached yet (LocalEntryNotFoundError): fetch the files the weight loader also uses
        root = hf_api().snapshot_download(model, **kwargs)
    return Path(root) / "transformer"


def _read_safetensors_header(path: Path) -> tuple[int, dict]:
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    return 8 + header_len, header


def mmap_modulation_tensors(transformer_dir: Path, num_blocks: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return ``[(weight, bias)] * num_blocks`` as zero-copy, read-only FP32 views of the checkpoint files."""
    files = sorted(transformer_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files in {transformer_dir}")
    found: dict[tuple[int, str], torch.Tensor] = {}
    for path in files:
        data_start, header = _read_safetensors_header(path)
        for name, meta in header.items():
            m = _MODULATION_RE.match(name)
            if m is None:
                continue
            if meta["dtype"] not in _SAFETENSORS_DTYPES:
                raise ValueError(f"{name} is {meta['dtype']}; host modulation expects F32 checkpoint tensors")
            begin, end = meta["data_offsets"]
            shape = tuple(meta["shape"])
            if (end - begin) != 4 * math.prod(shape):
                raise ValueError(f"{name}: data_offsets do not match shape {shape}")
            array = np.memmap(path, dtype=np.float32, mode="r", offset=data_start + begin, shape=shape)
            with warnings.catch_warnings():  # read-only mapping; the tensors are never written
                warnings.simplefilter("ignore", UserWarning)
                found[(int(m.group(1)), m.group(2))] = torch.from_numpy(array)
    tensors = []
    for i in range(num_blocks):
        if (i, "weight") not in found or (i, "bias") not in found:
            raise KeyError(f"transformer_blocks.{i}.norm1.linear weight/bias not found in {transformer_dir}")
        tensors.append((found[(i, "weight")], found[(i, "bias")]))
    if len(found) != 2 * num_blocks:
        raise ValueError(f"Expected {2 * num_blocks} modulation tensors, found {len(found)}")
    return tensors


class PAN2HostModulation:
    """Per-timestep modulation tables of all joint blocks, computed on the CPU from memory-mapped weights.

    Deliberately not an ``nn.Module``: the weight loader, layerwise offload, HSDP and ``.to(device)`` must never see
    the 17.7 GB of weights it holds.
    """

    def __init__(
        self,
        weights: Iterable[tuple[torch.Tensor, torch.Tensor]] | None = None,
        *,
        transformer_dir: Path | None = None,
        num_blocks: int,
        dim: int,
        cache_mb: int = 2048,
        num_threads: int | None = None,
    ) -> None:
        if weights is None and transformer_dir is None:
            raise ValueError("Pass either weights or transformer_dir")
        self.num_blocks = num_blocks
        self.dim = dim
        self._weights = list(weights) if weights is not None else None
        self._transformer_dir = transformer_dir
        entry_bytes = num_blocks * 6 * dim * 4
        self._max_entries = max(1, (cache_mb * 2**20) // entry_bytes)
        self._num_threads = num_threads or min(8, os.cpu_count() or 1)
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "computed": 0}

    # ------------------------------------------------------------------ weights
    @property
    def weights(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self._weights is None:
            self._weights = mmap_modulation_tensors(self._transformer_dir, self.num_blocks)
            logger.info(
                "PAN2 host modulation: mapped %d blocks of norm1.linear from %s (kept off the accelerator)",
                self.num_blocks,
                self._transformer_dir,
            )
        for w, b in self._weights:
            if w.shape != (6 * self.dim, self.dim) or b.shape != (6 * self.dim,):
                raise ValueError(f"Unexpected modulation shapes {tuple(w.shape)} / {tuple(b.shape)}")
        return self._weights

    # ------------------------------------------------------------------ compute
    @staticmethod
    def _key(t: float) -> int:
        return struct.unpack("<I", struct.pack("<f", float(t)))[0]

    def compute(self, temb: torch.Tensor) -> torch.Tensor:
        """``temb`` [S, dim] (any device) -> modulation [S, num_blocks, 6 * dim] FP32 on the CPU."""
        x = F.silu(temb.detach().to("cpu", torch.float32))
        out = torch.empty(x.shape[0], self.num_blocks, 6 * self.dim, dtype=torch.float32)
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(self._num_threads)
        try:
            for i, (w, b) in enumerate(self.weights):
                torch.addmm(b, x, w.t(), out=out[:, i])
        finally:
            torch.set_num_threads(prev_threads)
        self.stats["computed"] += x.shape[0]
        return out

    def _store(self, key: int, table: torch.Tensor) -> None:
        pinned = table.pin_memory() if torch.cuda.is_available() else table.clone()
        self._cache[key] = pinned
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)

    def prefetch(self, timesteps: torch.Tensor, temb: torch.Tensor) -> None:
        """Fill the cache for ``timesteps`` [S] whose embeddings ``temb`` [S, dim] the caller computed."""
        keys = [self._key(t) for t in timesteps.detach().float().cpu().tolist()]
        with self._lock:
            todo = [i for i, k in enumerate(keys) if k not in self._cache]
            if not todo:
                return
            tables = self.compute(temb[todo])
            for j, i in enumerate(todo):
                self._store(keys[i], tables[j])

    def lookup(self, timestep: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """Batch modulation: ``timestep`` [B], ``temb`` [B, dim] -> [B, num_blocks, 6 * dim] on ``temb``'s device."""
        values = timestep.detach().float().cpu().tolist()
        keys = [self._key(t) for t in values]
        rows: dict[int, torch.Tensor] = {}
        with self._lock:
            for i, k in enumerate(keys):
                if k in rows:
                    continue
                table = self._cache.get(k)
                if table is None:
                    self.stats["misses"] += 1
                    table = self.compute(temb[i : i + 1])[0]
                    self._store(k, table)
                    table = self._cache[k]
                else:
                    self.stats["hits"] += 1
                    self._cache.move_to_end(k)
                rows[k] = table
        unique = list(dict.fromkeys(keys))
        stacked = torch.stack([rows[k] for k in unique]) if len(unique) > 1 else rows[unique[0]].unsqueeze(0)
        device_table = stacked.to(temb.device, non_blocking=True)
        if len(unique) == 1:
            return device_table.expand(len(keys), -1, -1)
        index = torch.tensor([unique.index(k) for k in keys], device=temb.device)
        return device_table.index_select(0, index)
