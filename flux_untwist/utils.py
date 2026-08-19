from __future__ import annotations

import math
from typing import Any, Dict, List

import torch


def repeat_to_batch(x: torch.Tensor, batch: int) -> torch.Tensor:
    """Repeat a tensor along batch dim using ComfyUI's helper when available."""
    if int(x.shape[0]) == int(batch):
        return x
    try:
        import comfy.utils  # type: ignore

        return comfy.utils.repeat_to_batch_size(x, int(batch))
    except Exception:
        reps = math.ceil(int(batch) / max(1, int(x.shape[0])))
        return x.repeat((reps,) + (1,) * (x.ndim - 1))[: int(batch)]


def clone_model_options(options: Dict[str, Any]) -> Dict[str, Any]:
    """Copy mutable transformer option containers before adding patches."""
    out = dict(options or {})
    transformer_options = dict(out.get("transformer_options", {}) or {})
    patches = transformer_options.get("patches", {}) or {}
    patches_copy: Dict[str, List[Any]] = {}
    for key, value in patches.items():
        patches_copy[key] = list(value) if isinstance(value, list) else [value]
    transformer_options["patches"] = patches_copy
    out["transformer_options"] = transformer_options
    return out


def append_transformer_patch(model: Any, patch_name: str, patch_fn: Any) -> None:
    """Append a ComfyUI transformer patch without relying on optional ModelPatcher helpers."""
    model.model_options = clone_model_options(getattr(model, "model_options", {}) or {})
    transformer_options = model.model_options.setdefault("transformer_options", {})
    patches = transformer_options.setdefault("patches", {})
    patches.setdefault(patch_name, []).append(patch_fn)


def _scalar_timestep_value(timestep: Any) -> float:
    """Resolve a scalar sampling coordinate from ComfyUI timestep-like input."""
    if torch.is_tensor(timestep):
        return float(timestep.detach().float().mean().item())
    return float(timestep)


def sigma_from_timestep(timestep: Any) -> float:
    """Normalize Comfy timestep-like values into a sigma-like [0, 1] value."""
    try:
        value = _scalar_timestep_value(timestep)
        if not math.isfinite(value):
            return 1.0
        if 0.0 <= value <= 1.0:
            return max(0.0, min(1.0, value))
        if 1.0 < value <= 1000.0:
            return max(0.0, min(1.0, value / 1000.0))
    except Exception:
        pass
    return 1.0


def progress_from_timestep(timestep: Any) -> float:
    return max(0.0, min(1.0, 1.0 - sigma_from_timestep(timestep)))


def progress_from_schedule_index(timestep: Any, *, sigmas: Any = None) -> float:
    """Map the current denoiser call to its real sampler-schedule position.

    ComfyUI exposes the complete sampling schedule as
    ``transformer_options['sample_sigmas']``. K-diffusion-style schedules include
    a terminal zero that is a solver endpoint rather than a denoiser call, so that
    endpoint is excluded before converting an index to progress. The first model
    call maps to 0 and the final model call maps to 1.

    If no usable schedule is available, preserve the legacy scalar normalization
    path rather than guessing a schedule.
    """
    try:
        if torch.is_tensor(sigmas):
            schedule = sigmas.detach().float().flatten()
        elif isinstance(sigmas, (list, tuple)):
            schedule = torch.tensor(
                [_scalar_timestep_value(value) for value in sigmas],
                dtype=torch.float32,
            )
        else:
            schedule = None

        if schedule is not None and schedule.numel() >= 2:
            if not bool(torch.isfinite(schedule).all().item()):
                raise ValueError("sampling schedule contains non-finite values")

            # ComfyUI/K-diffusion schedules normally carry one terminal sigma=0
            # that is never passed to the denoiser. Removing it makes N denoiser
            # coordinates span indices 0..N-1 and therefore progress 0..1.
            if schedule.numel() >= 2 and abs(float(schedule[-1].item())) <= 1e-12:
                schedule = schedule[:-1]

            if schedule.numel() == 1:
                return 1.0
            if schedule.numel() >= 2:
                current = _scalar_timestep_value(timestep)
                if not math.isfinite(current):
                    raise ValueError("current sampling coordinate is non-finite")
                idx = int(torch.argmin((schedule - current).abs()).item())
                return max(0.0, min(1.0, idx / float(schedule.numel() - 1)))
    except Exception:
        pass

    return progress_from_timestep(timestep)


def _find_diffusion_model(model_patcher: Any, predicate: Any, error_message: str) -> Any:
    roots: List[Any] = []
    if hasattr(model_patcher, "model"):
        roots.append(model_patcher.model)
    roots.append(model_patcher)

    attr_paths = (
        "diffusion_model",
        "model.diffusion_model",
        "model.model.diffusion_model",
        "inner_model.diffusion_model",
        "model.inner_model.diffusion_model",
    )
    for root in roots:
        for path in attr_paths:
            obj = root
            ok = True
            for part in path.split("."):
                if not hasattr(obj, part):
                    ok = False
                    break
                obj = getattr(obj, part)
            if ok and predicate(obj):
                return obj

    seen = set()
    stack = list(roots)
    while stack and len(seen) < 512:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if predicate(obj):
            return obj
        for name in ("model", "inner_model", "diffusion_model", "unet", "wrapped"):
            if hasattr(obj, name):
                try:
                    stack.append(getattr(obj, name))
                except Exception:
                    pass
    raise RuntimeError(error_message)


def safe_get_diffusion_model(model_patcher: Any) -> Any:
    """Find the wrapped Flux diffusion model in common ComfyUI ModelPatcher layouts."""
    return _find_diffusion_model(
        model_patcher,
        _looks_like_flux,
        "Could not locate a Flux-like diffusion model on the supplied MODEL.",
    )


def safe_get_minimax_h3_model(model_patcher: Any) -> Any:
    """Find ComfyUI's native MiniMaxH3Model without assuming a fixed wrapper depth."""
    return _find_diffusion_model(
        model_patcher,
        _looks_like_minimax_h3,
        "Could not locate a native MiniMax H3 diffusion model on the supplied MODEL.",
    )


def _looks_like_flux(obj: Any) -> bool:
    return (
        hasattr(obj, "process_img")
        and hasattr(obj, "single_blocks")
        and hasattr(obj, "double_blocks")
        and hasattr(obj, "params")
    )


def _looks_like_minimax_h3(obj: Any) -> bool:
    if type(obj).__name__ == "MiniMaxH3Model":
        return True
    return (
        hasattr(obj, "blocks")
        and hasattr(obj, "rope")
        and hasattr(obj, "rope_freqs")
        and hasattr(obj, "audio_patch_proj")
        and hasattr(obj, "video_patch_proj")
        and hasattr(obj, "final_layer")
    )


def process_latent_for_model(model_patcher: Any, latent_samples: torch.Tensor) -> torch.Tensor:
    """Convert a ComfyUI LATENT tensor to model input latent space when available."""
    processor = None
    try:
        processor = getattr(getattr(model_patcher, "model", None), "process_latent_in", None)
    except Exception:
        processor = None
    if callable(processor):
        return processor(latent_samples)
    return latent_samples


def normalize_ref_latents(existing: Any) -> List[torch.Tensor]:
    if existing is None:
        return []
    if torch.is_tensor(existing):
        return [existing]
    if isinstance(existing, (list, tuple)):
        return [x for x in existing if torch.is_tensor(x)]
    return []
