# PAN 2 on RTX 4090 and 24 GB GPUs

This recipe runs [PAN2](PAN2.md) text-to-video and image-to-video on two 24 GB
GPUs (RTX 4090). It serves a pre-quantized FP8 transformer, keeps the FP32
adaLN modulation in host memory, runs the guardrails on the CPU, and swaps the
text encoder and the transformer on the GPUs with model-level CPU offload. For
one 48 GB GPU, see [PAN2-L40S](PAN2-L40S.md).

## Why the modulation matters

Each of the 44 joint blocks has an FP32 modulation linear `norm1.linear`
(4096 to 6 x 4096): 4.43B parameters, 17.7 GB, replicated on every
tensor-parallel rank. Its input depends only on the timestep. With host
modulation the linears are not built: their weights are memory-mapped from the
checkpoint, the per-timestep table (4.3 MB) is computed on the CPU, and each
forward uploads it. In eager mode the latents are bit-identical to the
in-model path. Without it, two 24 GB GPUs run out of memory at startup.

`model_config.pan2_host_modulation` defaults to `"auto"`, which turns host
modulation on when the modulation would take at least a quarter of a GPU's
memory and no layerwise offload streams the blocks. That covers 24 GB GPUs and
leaves the other PAN 2 profiles unchanged. Set it to `true` or `false` to force
it.

## Capacity

The profile is sized for 241 frames (10 s) at 704x1248, not the 481 frames of
the other PAN 2 profiles: 481 frames do not fit. Measured on 2x RTX 4090
(PCIe), TP=2, 50 steps, CFG 2.5, with GPU peer-to-peer enabled. Peak memory is
per GPU.

The standard NVIDIA driver does not enable peer-to-peer on GeForce GPUs; these
runs used the open kernel modules with a community P2P patch. Without
peer-to-peer, NCCL all-reduce at these activation sizes is about 6.7x slower
on the same machine, which adds roughly 8 minutes to a 121-frame clip
(estimated from all-reduce timings, not measured end to end). Check with
`nvidia-smi topo -p2p r`.

| Output | Time | Peak memory |
| --- | ---: | ---: |
| 121 frames (5 s) | 9.5 min | 19.4 GiB |
| 241 frames (10 s) | 29 min | 22.9 GiB |

The checkpoint is
[`Eisenhawer/PAN2-FP8`](https://huggingface.co/Eisenhawer/PAN2-FP8), a
community conversion of `IFM/PAN2`: the 528 joint-block attention and FFN
linears are FP8 E4M3 with per-channel scales (ModelOpt
`FP8_PER_CHANNEL_PER_TOKEN`, detected from its config), and every other tensor
is unchanged. Its model card reports single-step fidelity against BF16.

Host RAM: 64 GB or more. A request is one RPC to the workers and the
synchronous HTTP endpoint waits for it, so both 600 s timeouts must be raised.

## 2 GPUs

```bash
pip install git+https://github.com/ifm-ai/pan2-guardrail  # PAN 2 guardrails
export VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=7200 VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200
vllm serve Eisenhawer/PAN2-FP8 --omni --port 8091 \
  --num-gpus 2 --tensor-parallel-size 2 \
  --enable-cpu-offload \
  --vae-use-tiling --vae-patch-parallel-size 2 --vae-parallel-mode tile \
  --enforce-eager \
  --stage-overrides '{"0": {"model_config": {"guardrail_device": "cpu"}}}'
```

To quantize the original weights while loading instead, serve `IFM/PAN2` with
`--quantization convrot_int8`. It loads in about 130 s against 85 s, and its
241-frame run took 34 min.

Send requests as shown in [PAN2](PAN2.md#requests).
