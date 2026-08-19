# ComfyUI Untwisting RoPE v0.2.0

v0.2.0 adds native MiniMax H3 support while preserving the existing Flux/Flux.2 path. The H3 implementation incorporates multi-chunk runtime findings, exact sampler-window semantics, source-kind-aware reference scoping, and Spectrum interoperability.

## Native MiniMax H3 support

`MiniMax H3 Untwist RoPE` integrates with ComfyUI's native H3 reference conditioning and applies frequency-aware modulation after native Q/K normalization + split-half RoPE and before attention.

The H3 path leaves queries, values, text, guide/keyframe rows, reference audio, target audio, target video and the unrotated head tail unchanged.

## Actual sampler-schedule progress

H3 denoising progress is derived from ComfyUI's complete `transformer_options["sample_sigmas"]` schedule. The terminal zero solver endpoint is excluded because it is not a denoiser call, then the current call is matched to the remaining schedule positions.

A 19-call run therefore maps the first denoiser call to progress `0.0` and the final denoiser call to `1.0`. `end_percent=0.90` genuinely disables Untwist for the final portion of sampling.

The legacy scalar `timestep` normalization remains only as a fallback when a real sampler schedule is unavailable.

## Correct native-reference scoping

Native H3 uses the same packed segment label, `ref_img`, for visual rows from image, video and video+audio references. Untwist pairs `minimax_payload["refs"]` with `layout.segments` in native order and fails closed on any count mismatch.

The `reference_scope` control is:

- `image_and_video` — **default**; selects native `image` and pure `video`, excludes `video_audio`, and protects Continuum carry-over context;
- `image_only` — selects only native `image` references;
- `all_visual_including_continuum` — explicit advanced/experimental opt-in for `image`, `video`, `video_audio`, and Continuum context.

H3 Continuum's `_h3_continuum` metadata and legacy context marker remain recognized. A marked Continuum reference is protected under the safe scopes even when the carry-over is video-only.

## Temporal-axis policy

Current H3 uses three rotary banks (`t`, `h`, `w`), 16 frequencies per bank and 128-dimensional attention heads. Split-half RoPE rotates 96 channels and leaves 32 channels unrotated.

The paper analyzes spatial reference behavior. The H3 default of keeping the temporal `t` bank native while scaling only `h` and `w` is an architecture/runtime choice. `scale_temporal_axis=true` remains an explicit experimental opt-in.

## H3 release defaults

Current H3 generation testing uses this release starting point while Flux defaults remain unchanged:

```text
high_scale_start    = 0.95
high_scale_end      = 1.05
low_scale_start     = 1.00
low_scale_end       = 1.05
beta                = 2.0
start_percent       = 0.0
end_percent         = 0.90
reference_scope     = image_and_video
scale_temporal_axis = false
```

`beta=2` follows the paper's experiments. The H3 scale/window/scope values are empirical runtime defaults rather than an H3-optimal paper claim.

## Spectrum external-patch profile

Untwist publishes a namespaced H3 profile for Spectrum with provider/instance identity, kind `visual_reference_attention_modulation`, block coverage, active progress window, hard-boundary flags, reference scope, high/low endpoint scales, `beta`, temporal-axis mode, and a scalar strength summary.

Per-call runtime metadata exposes schedule progress and active state. The separate profile namespace keeps older Spectrum versions backward-compatible: they ignore the new keys instead of rejecting an unsupported kind through the existing Diff-Aid-only schema.

The companion Spectrum change recognizes this profile alongside Diff-Aid and forces an actual call when Untwist crosses a hard boundary such as the default 90% cutoff.

## Exact no-op and inactive-call behavior

If all four H3 scale endpoints are `1.0`, the node returns an exact model-patch no-op: it installs no model wrapper, attention override, or Spectrum profile.

For a non-neutral configuration, the optimized-attention override is attached only when the actual schedule window is active, native ref mapping is valid and at least one reference is selected by scope. No-ref, protected-ref, inactive-window and ambiguous-mapping calls stay on the native attention path.

When another extension supplies an `optimized_attention_override` at call time, Untwist composes with that call-local override. A patch-time override is only the fallback when the call has no override of its own.

## Flux/Flux.2 behavior

The Flux implementation remains architecture-specific and unchanged in behavior. It continues to use ComfyUI's `attn1_patch` integration, native Flux reference-latent handling, the paper-aligned starting scales, optional block range and optional Q/K AdaIN.

## Tests and release automation

Regression coverage includes:

- existing Flux frequency/range behavior;
- H3 split-half mapping and the 32-channel unrotated tail;
- native temporal-bank preservation and explicit temporal opt-in;
- H3 geometry validation errors;
- exact 19-call schedule-index progress and final `1.0` mapping;
- `end_percent=0.90` final-call deactivation;
- exact `image_only`, `image_and_video`, and all-visual kind filtering;
- both public `minimax_h3_visual_reference_ranges` call shapes;
- `video_audio` exclusion from the safe default;
- H3 Continuum carry-over exclusion and explicit opt-in;
- fail-closed ref/range mismatch behavior;
- no-reference native-attention no-op behavior;
- reference-only K scaling and Q/V/input-K preservation;
- exact all-ones model no-op behavior;
- call-time optimized-attention/model-wrapper composition;
- model-options clone isolation;
- Spectrum static/runtime profile emission.

The pinned native-ComfyUI H3 CI contract also validates retention of the native ref list in `minimax_payload`, because safe H3 ref-to-row pairing depends on that runtime data.

CI runs the Python unit/quality matrix and native H3 contract on pull requests and main. A successful current-main test run creates the GitHub release archive and SHA-256 checksum, then explicitly dispatches registry publication for that release tag. The registry workflow validates that the supplied tag is a published GitHub release and that its version matches `pyproject.toml` before publishing.
