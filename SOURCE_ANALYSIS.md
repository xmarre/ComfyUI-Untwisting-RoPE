# Source analysis and design decisions

## Paper behavior to preserve

The paper's core mechanism is a targeted shared-attention intervention:

1. RoPE decomposes into frequency components with different positional sensitivity.
2. High-frequency components are strongly local and can encourage reference-content copying.
3. Lower-frequency components preserve broader/global reference association.
4. The intervention modulates reference image **keys**, not target/text keys and not values.
5. The frequency scale is smooth across the RoPE spectrum and schedulable over denoising.
6. The paper uses `s_d = s_hf + (s_lf - s_hf) * d_norm^beta`, with `beta=2` in its experiments.

The paper validates Flux image generation. MiniMax H3 support is an architecture-specific adaptation; H3 defaults and video/reference-scope behavior are runtime engineering choices.

## Flux path

Flux remains separate from H3:

- native Flux `ref_latents` are used;
- ComfyUI `attn1_patch` is used instead of replacing attention modules;
- reference ranges are derived from Flux `img_slice` and `reference_image_num_tokens`;
- the existing single-stream block controls and optional Q/K AdaIN remain available;
- failure to identify Flux reference ranges is a no-op.

The Flux path keeps its original defaults and behavior.

## Native MiniMax H3 architecture

Native ComfyUI H3 is a packed single-stream audio/video DiT. Its logical order is:

```text
[text | reference blocks | target audio | target video]
```

`PackedLayout.segments` distinguishes row types such as `ref_img` and `ref_audio`. `ref_img` is a row type rather than a native-reference identity:

- `kind="image"` -> one `ref_img` block;
- `kind="video"` -> one `ref_img` block;
- `kind="video_audio"` -> optional `ref_audio` followed by one `ref_img` block.

ComfyUI retains the original ref dictionaries in `minimax_payload["refs"]` while also building the packed layout. Untwist therefore uses:

- reference identity/kind/metadata from `payload["refs"]`;
- exact packed row ranges from `payload["layout"].segments`.

## H3 reference-to-row ownership

The H3 node does not create, encode, resize or reorder native references. For every call it:

1. enumerates visual refs whose kind is `image`, `video` or `video_audio`;
2. enumerates packed `ref_img` ranges;
3. requires the counts to match exactly;
4. pairs the lists one-for-one in native order;
5. applies the selected source-kind scope;
6. fails closed with no K modulation when ownership cannot be proven.

### Reference scopes

The supported scopes have exact source-kind semantics:

```text
image_only                    -> image
image_and_video               -> image + pure video
all_visual_including_continuum -> image + video + video_audio
```

`image_and_video` is the H3 default. It deliberately excludes mixed `video_audio` references.

Continuum/native-RoPE protection is applied in addition to kind filtering. A reference carrying Continuum preservation metadata stays native under `image_only` and `image_and_video`, including a video-only Continuum carry-over. `all_visual_including_continuum` is the explicit opt-in that lifts that protection.

## H3 Continuum interoperability

H3 Continuum publishes namespaced metadata on its carry-over ref:

```python
ref["_h3_continuum"] = {
    "api": 1,
    "role": "video_context",
    "audio_role": "audio_context" or None,
    "preserve_rope": True,
}
```

Untwist recognizes `preserve_rope`, the Continuum `video_context` role, and the legacy `_h3cj_video_context` marker.

The Continuum companion PR also regression-tests that `image`, `video`, and `video_audio` labels remain distinct in `minimax_refs`, which native H3 subsequently preserves in `minimax_payload["refs"]`.

## Root cause of the H3 progress-window bug

The H3 wrapper previously used generic scalar normalization:

```python
progress = progress_from_timestep(timestep)
```

That path assumes a normalized sigma or a `0..1000` timestep scalar. MiniMax H3's actual sampler calls do not satisfy the intended linear-progress interpretation, so a 19-step run could end around `progress ~= 0.845`. `end_percent=0.90` then remained active on the final transformer call.

The relevant ComfyUI sampling path provides two different pieces of state:

- `transformer_options["sample_sigmas"]` is the **complete sampler schedule**;
- `transformer_options["sigmas"]` is the **current call coordinate**.

`model_function_wrapper` receives the current timestep and conditioning dictionary; it does not normally receive the full schedule as `args["sigmas"]`.

The H3 fix therefore reads `c["transformer_options"]["sample_sigmas"]`. An optional `args["sigmas"]` fallback is accepted for alternate wrappers. The model's `model_sampling.sigmas` table is not used because it is not the active sampler schedule.

### Schedule-index mapping invariant

K-diffusion-style ComfyUI schedules normally contain `N+1` sigma values for `N` denoiser calls; the final zero is a solver endpoint and is never passed to the denoiser. Untwist removes that terminal zero and matches the current call to the nearest remaining schedule coordinate:

```text
first denoiser call -> index 0     -> progress 0.0
final denoiser call -> index N - 1 -> progress 1.0
```

The generic scalar path is retained only when no valid sampler schedule is available.

This restores the required window invariant: with `end_percent=0.90`, calls in the final 10% are inactive and do not install the H3 attention override.

## H3 RoPE geometry

Current native H3 uses:

- 56 attention heads;
- `head_dim = 128`;
- `rope.inv_freq` with 16 frequencies;
- three rotary coordinate axes: `t`, `h`, `w`;
- split-half RoPE over `2 * 3 * 16 = 96` channels;
- a 32-channel non-rotary tail.

The native split-half layout is treated as:

```text
half    = [t16 | h16 | w16]
rotated = [half | half]
full    = [rotated | tail32]
```

The scale vector is mirrored over the paired half and the 32-channel tail remains exactly `1`.

## Temporal-axis decision

The paper analyzes spatial reference behavior. H3 keeps its temporal `t` rotary bank exactly native by default as an architecture/runtime policy:

```text
scale_temporal_axis = false
```

For the current three-axis layout, `h` and `w` receive the frequency schedule, the paired split half mirrors those choices, and the 32-channel unrotated tail remains `1`. `scale_temporal_axis=true` is an explicit experiment.

## Attention intervention point

H3 performs Q/K RMSNorm and split-half RoPE before optimized attention. The patch composes an `optimized_attention_override` and changes selected K rows after native RoPE and before attention.

When active:

- only selected native-reference K rows are scaled;
- Q is unchanged;
- V is unchanged;
- the source K tensor is cloned before non-neutral modifications;
- ref audio is unchanged;
- text, guide/keyframe, target audio and target video rows are unchanged.

## Exact and call-local no-op behavior

A neutral H3 configuration with all four scale endpoints equal to `1.0` returns a cloned MODEL without installing the Untwist model wrapper, attention override, or Spectrum profile.

For a non-neutral node, a model call stays on native optimized attention when:

- actual schedule progress is outside the configured window;
- no references are selected by scope;
- all available refs are protected;
- refs and `ref_img` ranges cannot be reconciled.

Within the attention override, an interpolated all-ones scale returns the original K tensor by identity.

## H3 defaults

The current H3 release defaults are:

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

The high-frequency schedule moves from `0.95 -> 1.00` while the low-frequency schedule uses a small `1.00 -> 1.05` lift. These are empirical H3 release settings from generation testing, not a paper-derived H3 optimum. `beta=2` follows the paper.

## Spectrum external-patch contract

Spectrum's existing external-patch schema is intentionally strict and currently represents Diff-Aid's `text_activation_modulation`. Publishing Untwist under that key with a new kind would make older/current Spectrum builds reject the contract and fail safe to all-actual sampling.

Untwist therefore publishes a separate, backward-compatible namespace:

```text
spectrum_h3_visual_reference_patch_profiles
spectrum_h3_visual_reference_patch_runtime
```

The static profile includes:

- schema/provider/instance identity;
- `kind = visual_reference_attention_modulation`;
- H3 block coverage;
- progress start/end;
- hard-start/hard-end flags;
- source-kind scope;
- high/low scale start/end;
- `beta`;
- temporal-axis mode;
- scalar strength summary = largest absolute endpoint distance from `1.0`.
- v2 terminal-PECE eligibility capability.

Per-call runtime metadata includes actual schedule progress and the call-local active flag.

A companion Spectrum PR recognizes this namespace, validates it independently, folds producer-declared hard progress boundaries into Spectrum's existing transaction guard, preserves Diff-Aid's schema, and includes the full Untwist strength metadata in the external-profile fingerprint. Older Spectrum versions ignore the separate keys.

With Diff-Aid and Untwist stacked, the recognized runtime kinds become:

```text
text_activation_modulation,visual_reference_attention_modulation
```

The default Untwist `end_percent=0.90` boundary is hard because the configured end scales are non-neutral. The boundary is schedule semantics rather than protection from a sigma-zero singularity, so changing it to `1.0` would be a model-behavior change. Visual-profile schema v2 preserves the boundary and declares `terminal_pece_exact_corrector_safe=true` only for weak (`strength <= 0.05`), late (`end_percent >= 0.90`), spatial-only safe-scope profiles. Spectrum remains responsible for proving the terminal same-outer PECE topology and exact corrected phase; every other case retains the actual promotion.

## Wrapper/override composition and mutation safety

The H3 patch:

- clones model options before mutation;
- preserves an existing model-function wrapper and forwards the complete argument dictionary;
- composes the call-time `optimized_attention_override` when present, using the patch-time override only as a fallback;
- keeps selected row ranges and runtime metadata call-local;
- appends Spectrum profile/runtime state without mutating the source MODEL's option dictionaries;
- uses no global mutable reference state.

Recommended chain:

```text
H3 -> Diff-Aid -> Untwist RoPE -> Spectrum -> sampler / Continuum
```

## Deliberately not implemented

- RF inversion or doubled reference batches.
- Z-Image/NextDiT integration.
- H3 reference creation/encoding inside the model patch.
- H3 audio-reference frequency modulation.
- H3 block monkey-patching.
- automatic Untwisting of mixed `video_audio` references under the safe default.
- automatic Untwisting of Continuum carry-over references.
- automatic temporal-axis Untwisting.
- guessing a native reference mapping when `refs` and packed `ref_img` rows disagree.

## Tests

Pure unit coverage includes:

- existing Flux behavior;
- H3 split-half geometry and unrotated tail;
- native temporal-bank preservation and explicit temporal opt-in;
- H3 geometry validation errors;
- 19-call schedule-index mapping with final progress exactly `1.0`;
- `end_percent=0.90` deactivation for final calls;
- exact source-kind filtering for all three reference scopes;
- both public H3 reference-range helper call shapes;
- explicit exclusion of ordinary `video_audio` under `image_and_video`;
- Continuum context protection and explicit opt-in;
- fail-closed ref/range mismatch behavior;
- no-reference native-attention no-op behavior;
- selected-reference-only K scaling;
- exact all-ones no-op;
- Q/V/input-K preservation;
- call-time optimized-attention override composition;
- model-wrapper argument preservation;
- model-options clone isolation;
- Spectrum static/runtime profile emission.

CI pins a reviewed ComfyUI revision and checks the native H3 contracts the adapter depends on, including retention of `payload["refs"]`.

## Remaining empirical work

- Continue fixed-seed H3 quality/identity/copying comparisons around the release defaults.
- Compare ordinary image and pure-video reference behavior under the release defaults.
- Evaluate the advanced `video_audio`/Continuum opt-in separately.
- Evaluate whether temporal-axis scaling has a useful niche without harming motion/continuity.
- Measure active-call attention-backend overhead in real workflows.
