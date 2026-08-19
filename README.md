# ComfyUI Untwisting RoPE

Frequency-aware RoPE reference-attention modulation for Flux/Flux.2 and native MiniMax H3 in ComfyUI.

This package implements the reference-key frequency intervention from:

Aryan Mikaeili, Or Patashnik, Andrea Tagliasacchi, Daniel Cohen-Or, Ali Mahdavi-Amiri, **“Untwisting RoPE: Frequency Control for Shared Attention in DiTs,”** arXiv:2602.05013, 2026.

The paper shows that high-frequency RoPE components strongly encode locality and can make shared reference attention copy reference structure/content, while lower-frequency components preserve broader semantic/reference association. The core intervention scales only reference **keys** by RoPE frequency; target/text keys, queries and values are unchanged by the basic method.

## Nodes

### Flux.2 Untwist RoPE

The Flux node keeps the original implementation path:

- uses ComfyUI's Flux `attn1_patch` hook;
- works with a supplied `reference_latent` or native `ref_latents` conditioning;
- scales only reference-image K channels in Flux single-stream attention;
- supports the existing block range, denoising window, reference-position method and optional Q/K AdaIN controls.

Flux defaults remain paper-aligned:

```text
high_scale_start = 0.25
high_scale_end   = 0.75
low_scale_start  = 1.00
low_scale_end    = 1.40
beta             = 2.0
start_percent    = 0.0
end_percent      = 1.0
```

### MiniMax H3 Untwist RoPE

The H3 node uses ComfyUI's native MiniMax H3 ref2va path. It does not encode or append a second reference representation.

Native H3 packs text, references, target audio and target video into one sequence. `PackedLayout.segments` labels the visual rows of **image**, **video**, and **video+audio** references as `ref_img`, so `ref_img` alone does not identify the source reference kind.

The H3 implementation pairs native `minimax_payload["refs"]` entries with the `ref_img` segments in order and fails closed if the mapping is ambiguous. Source kind remains authoritative for scope selection.

#### H3 reference scope

`reference_scope` controls which native visual references may be modified:

- `image_and_video` — **default**. Includes ordinary native `image` and pure `video` references. Mixed `video_audio` references are excluded. H3 Continuum carry-over context remains protected.
- `image_only` — includes only native `image` references.
- `all_visual_including_continuum` — includes `image`, `video`, and `video_audio`, including Continuum context. This is advanced/experimental.

H3 Continuum publishes `_h3_continuum` metadata on its carry-over ref. The Untwist node recognizes both the current `preserve_rope=true` contract and the existing Continuum context role/legacy marker. A Continuum carry-over remains native under the safe scopes even when it is video-only.

#### H3 denoising progress

H3 windowing uses the **actual sampler schedule position**. Current ComfyUI publishes the full schedule as `transformer_options["sample_sigmas"]`; the per-call `transformer_options["sigmas"]` value contains only the current coordinate and is not used as the schedule.

K-diffusion-style schedules include a terminal zero endpoint that is not a denoiser call. Untwist excludes that endpoint, matches the current call to the remaining schedule, and maps the first/final denoiser calls to progress `0.0`/`1.0`.

This makes `end_percent=0.90` a real hard cutoff. On a 19-call schedule, calls after 90% progress stay on native H3 attention, including the final call at progress `1.0`.

#### H3 RoPE geometry

Current native H3 uses:

```text
head_dim = 128
axes     = t, h, w
freqs    = 16 per axis
rotated  = 2 * 3 * 16 = 96 channels
native/unrotated tail = 32 channels
```

H3's split-half layout is:

```text
half    = [t16 | h16 | w16]
rotated = [half | half]
full    = [rotated | native tail32]
```

The unrotated tail always remains `1.0`.

The paper analyzes spatial reference behavior. For MiniMax H3, keeping the `t` rotary bank native and applying the frequency schedule only to `h` and `w` is an architecture/runtime design choice. `scale_temporal_axis=true` is the explicit experimental opt-in for t/h/w scaling.

#### H3 starting settings

Current H3 generation testing uses this restrained release default:

```text
high_scale_start    = 0.95
high_scale_end      = 1.00
low_scale_start     = 1.00
low_scale_end       = 1.05
beta                = 2.0
start_percent       = 0.0
end_percent         = 0.90
reference_scope     = image_and_video
scale_temporal_axis = false
```

These are empirical H3 starting values, not a paper-derived H3 optimum. `beta=2` follows the paper; the H3 scale/window/scope defaults are architecture/runtime-specific choices.

## Spectrum external-patch profile

The H3 node publishes a namespaced Spectrum runtime profile containing:

- provider and instance ID;
- kind `visual_reference_attention_modulation`;
- H3 block coverage;
- progress window and hard-boundary flags;
- reference scope;
- high/low scale start/end values;
- `beta` and temporal-axis mode;
- a scalar strength summary derived from the largest endpoint distance from `1.0`.

Per-call runtime metadata publishes actual schedule progress and whether Untwist is active for that call. The profile uses separate keys from Diff-Aid so older Spectrum releases simply ignore it rather than rejecting a foreign kind through the Diff-Aid-only schema.

Spectrum support for this profile is implemented in the companion Spectrum PR. With Diff-Aid and Untwist both active, Spectrum can recognize two external patch kinds and force an actual call at Untwist's hard `end_percent` transition.

## Exact no-op behavior

For H3, setting all four scale endpoints to `1.0` returns an exact model-patch no-op. The node does not install its model wrapper, optimized-attention override, or Spectrum profile.

When the configured schedule is non-neutral and a specific model call has no selected refs, an inactive denoising window, or an ambiguous native-ref/segment mapping, the per-call optimized-attention override is not installed. The native attention path remains in control.

## H3 workflow placement

Recommended MODEL chain when also using Diff-Aid and Spectrum:

```text
MiniMax H3 model
  -> MiniMax H3 Diff-Aid Sparse Patch
  -> MiniMax H3 Untwist RoPE
  -> Spectrum Apply MiniMax H3
  -> sampler / H3 Continuum
```

Continuum generates its next-chunk carry-over reference inside the sampling workflow; the Untwist node detects and excludes that reference at runtime under the safe scopes.

## Tuning

For H3, change one dimension at a time:

- baked/plasticky appearance: move `high_scale_start` toward `1.0` and keep low-frequency gain small;
- excessive global reference pull: reduce `low_scale_end` toward `1.0`;
- insufficient reduction of reference pose/composition copying: lower `high_scale_start` gradually;
- late-stage roughness: reduce `end_percent`;
- only test `scale_temporal_axis=true` after the spatial-only configuration is known-good;
- use `image_only` when pure video references should also remain native;
- use `all_visual_including_continuum` only as an explicit experiment.

A neutral control with all scales `1.0` is the first diagnostic if a new artifact appears.

## Implementation details

### Flux

- Flux reference ranges come from `img_slice` and `reference_image_num_tokens`.
- Pair-wise scaling is applied through the native Flux attention patch path.
- Existing Flux behavior is intentionally unchanged in v0.2.0.

### MiniMax H3

- native reference identity/kind comes from `minimax_payload["refs"]`;
- exact row ranges come from `minimax_payload["layout"].segments`;
- native visual ref order is paired one-for-one with `ref_img` ranges;
- any count mismatch fails closed;
- `image_and_video` selects only `image` + pure `video`;
- `video_audio` requires `all_visual_including_continuum`;
- Continuum context requesting native RoPE is excluded under the safe scopes;
- only selected K rows are cloned/scaled;
- Q, V, text, guides/keyframes, target audio/video and `ref_audio` are unchanged;
- scaling happens after H3's fused RMSNorm + split-half RoPE and before attention via `optimized_attention_override`;
- the call-time `optimized_attention_override` is composed when present, with the patch-time override used only as a fallback;
- H3 `patches_replace["dit"]` remains unclaimed.

## Validation

The unit suite covers:

- existing Flux frequency/range behavior;
- H3 split-half mapping and the 32-channel native tail;
- native temporal-bank preservation and explicit temporal opt-in;
- H3 geometry validation failures;
- 19-call schedule-index progress with final progress exactly `1.0`;
- real `end_percent=0.90` deactivation on final calls;
- exact `image_only`, `image_and_video`, and `all_visual_including_continuum` kind semantics;
- both public H3 reference-range helper call shapes;
- Continuum carry-over exclusion and explicit opt-in;
- fail-closed behavior when refs and packed ranges do not reconcile;
- no-reference native-attention no-op behavior;
- reference-only K scaling;
- exact all-ones no-op;
- Q/V/input-K preservation;
- call-time optimized-attention override composition;
- model-wrapper composition;
- cloned-model-options isolation;
- Spectrum static/runtime profile emission.

CI also checks the pinned native ComfyUI H3 contract, including preservation of the `minimax_payload["refs"]` list required for safe ref-to-row pairing.

## Compatibility and risks

- MiniMax H3 support targets ComfyUI's native H3 implementation and its current three-axis split-half RoPE geometry.
- Ordinary native pure-video Untwisting is supported by default; H3-specific video quality still requires empirical tuning.
- Mixed `video_audio` and Continuum carry-over Untwisting are advanced opt-ins.
- Temporal-axis Untwisting remains experimental and disabled by default.
- Active `optimized_attention_override` calls can materialize `AttentionTensorContainer` inputs before the custom override. Backends with specialized container fast paths may have a performance difference on active calls.

## Installation

Clone into `ComfyUI/custom_nodes` and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xmarre/ComfyUI-Untwisting-RoPE
```

No dependency beyond a working ComfyUI/PyTorch environment is added by this node.

## License

MIT. The paper, ComfyUI and model weights remain under their own licenses.
