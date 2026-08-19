# ComfyUI Untwisting RoPE v0.2.1

v0.2.1 completes the project-wide rebrand now that Untwisting RoPE supports both Flux/Flux.2 and native MiniMax H3.

## Generic project branding

- GitHub repository branding is now `ComfyUI-Untwisting-RoPE` rather than the old Flux.2-specific name.
- The README title and installation command use the generic project name.
- Project URLs in `pyproject.toml` point to the generic repository name.
- GitHub release archives are now named `ComfyUI-Untwisting-RoPE-v<version>.zip` and use the same generic root directory inside the archive.
- The Comfy Registry display name explicitly lists Flux/Flux.2 and MiniMax H3 support.

## Registry identity compatibility

The existing `[project].name = "comfyui-flux2-untwisting-rope"` value is intentionally retained. Comfy Registry defines the project name as the immutable node ID after first publication, so changing it would create/break registry identity instead of performing a display rename. Repository URLs and `DisplayName` carry the new generic branding while preserving updates for existing installations.

## Runtime behavior

There are no model/runtime behavior changes in v0.2.1. The MiniMax H3 defaults introduced in v0.2.0 remain:

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

Flux/Flux.2 behavior is unchanged.
