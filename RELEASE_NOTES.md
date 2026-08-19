# ComfyUI Untwisting RoPE v0.2.2

v0.2.2 corrects the MiniMax H3 release defaults so the shipped node matches the tested configuration.

## Correct MiniMax H3 defaults

The exact H3 defaults are now:

```text
high_scale_start    = 0.95
high_scale_end      = 1.00
low_scale_start     = 1.00
low_scale_end       = 1.05
beta                = 2.0
start_percent       = 0.0
end_percent         = 0.90
verbose             = false
reference_scope     = image_and_video
scale_temporal_axis = false
```

The previous release incorrectly shipped `high_scale_end = 1.05`. This release corrects that value to `1.00` everywhere it can affect or describe the H3 default contract:

- ComfyUI node UI default;
- H3 input sanitization fallback;
- attention-helper fallback values when a runtime configuration omits scale keys;
- Spectrum profile regression expectations;
- README and source-analysis documentation;
- explicit regression coverage for the complete release-default set and fallback schedule endpoints.

Flux/Flux.2 behavior and defaults are unchanged.

## Project branding

The generic `ComfyUI-Untwisting-RoPE` repository/display branding introduced in v0.2.1 is retained. The legacy `[project].name = "comfyui-flux2-untwisting-rope"` value remains intentionally unchanged because it is the existing Comfy Registry node ID.
