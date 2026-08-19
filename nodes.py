from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

try:
    from .flux_untwist.config import (
        _H3_PREFIX,
        _PREFIX,
        MiniMaxH3UntwistConfig,
        UntwistConfig,
        clamp_float,
        clamp_int,
        coerce_bool,
        normalize_percent_window,
        normalize_reference_method,
        safe_axes_dim,
        schedule_fraction,
    )
    from .flux_untwist.patches import (
        flux_untwist_attn1_patch,
        make_minimax_h3_attention_override,
        minimax_h3_visual_reference_selection,
    )
    from .flux_untwist.spectrum_h3 import (
        append_spectrum_h3_runtime,
        register_spectrum_h3_profile,
    )
    from .flux_untwist.utils import (
        append_transformer_patch,
        clone_model_options,
        normalize_ref_latents,
        process_latent_for_model,
        progress_from_schedule_index,
        progress_from_timestep,
        repeat_to_batch,
        safe_get_diffusion_model,
        safe_get_minimax_h3_model,
    )
except ImportError:
    from flux_untwist.config import (
        _H3_PREFIX,
        _PREFIX,
        MiniMaxH3UntwistConfig,
        UntwistConfig,
        clamp_float,
        clamp_int,
        coerce_bool,
        normalize_percent_window,
        normalize_reference_method,
        safe_axes_dim,
        schedule_fraction,
    )
    from flux_untwist.patches import (
        flux_untwist_attn1_patch,
        make_minimax_h3_attention_override,
        minimax_h3_visual_reference_selection,
    )
    from flux_untwist.spectrum_h3 import (
        append_spectrum_h3_runtime,
        register_spectrum_h3_profile,
    )
    from flux_untwist.utils import (
        append_transformer_patch,
        clone_model_options,
        normalize_ref_latents,
        process_latent_for_model,
        progress_from_schedule_index,
        progress_from_timestep,
        repeat_to_batch,
        safe_get_diffusion_model,
        safe_get_minimax_h3_model,
    )


class Flux2UntwistRoPE:
    """Patch FLUX/FLUX.2 reference attention with frequency-aware RoPE key scaling."""

    CATEGORY = "model_patches/Flux2 Untwisting RoPE"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    DESCRIPTION = (
        "Adds reference latents to a Flux/Flux.2 model and scales only reference image keys "
        "in single-stream attention according to RoPE frequency bands."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "high_scale_start": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": -4.0,
                        "max": 8.0,
                        "step": 0.01,
                        "tooltip": "Scale for highest-frequency reference-key RoPE channels at the start of the active window. <1 reduces positional copying.",
                    },
                ),
                "high_scale_end": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": -4.0,
                        "max": 8.0,
                        "step": 0.01,
                        "tooltip": "Scale for highest-frequency reference-key RoPE channels at the end of the active window.",
                    },
                ),
                "low_scale_start": (
                    "FLOAT",
                    {
                        "default": 1.00,
                        "min": -4.0,
                        "max": 8.0,
                        "step": 0.01,
                        "tooltip": "Scale for lowest-frequency reference-key RoPE channels at the start of the active window.",
                    },
                ),
                "low_scale_end": (
                    "FLOAT",
                    {
                        "default": 1.40,
                        "min": -4.0,
                        "max": 8.0,
                        "step": 0.01,
                        "tooltip": "Scale for lowest-frequency reference-key RoPE channels at the end of the active window. >1 increases global style/reference pull.",
                    },
                ),
                "beta": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.01,
                        "max": 32.0,
                        "step": 0.01,
                        "tooltip": "Polynomial interpolation exponent from high-frequency to low-frequency scales. Paper default is 2.",
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "start_single_block": (
                    "INT",
                    {"default": 0, "min": 0, "max": 999, "step": 1},
                ),
                "end_single_block": (
                    "INT",
                    {"default": 999, "min": 0, "max": 999, "step": 1},
                ),
                "reference_latents_method": (
                    ["index", "offset", "uxo", "index_timestep_zero"],
                    {
                        "default": "index",
                        "tooltip": "How Flux positions appended reference latent tokens. index keeps the spatial grid aligned and differs only by image index; offset/uxo use spatial offsets.",
                    },
                ),
                "qk_adain_strength": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Optional StyleAligned-style AdaIN on target image Q/K statistics. 0 preserves native Flux.2 behavior.",
                    },
                ),
                "verbose": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "reference_latent": (
                    "LATENT",
                    {
                        "tooltip": "Encoded reference image latent. You may leave this unconnected if your conditioning already supplies ref_latents.",
                    },
                ),
            },
        }

    def patch(
        self,
        model: Any,
        high_scale_start: float,
        high_scale_end: float,
        low_scale_start: float,
        low_scale_end: float,
        beta: float,
        start_percent: float,
        end_percent: float,
        start_single_block: int,
        end_single_block: int,
        reference_latents_method: str,
        qk_adain_strength: float,
        verbose: bool = False,
        reference_latent: Optional[Dict[str, Any]] = None,
    ):
        node_verbose = coerce_bool(verbose)
        start_percent, end_percent = normalize_percent_window(start_percent, end_percent)
        start_single_block = clamp_int(start_single_block, 0, 999, 0)
        end_single_block = clamp_int(end_single_block, 0, 999, 999)
        if end_single_block < start_single_block:
            start_single_block, end_single_block = end_single_block, start_single_block

        ref_samples_cpu: Optional[torch.Tensor] = None
        if isinstance(reference_latent, dict) and torch.is_tensor(reference_latent.get("samples")):
            samples = reference_latent["samples"].detach()
            if samples.ndim != 4:
                raise RuntimeError(f"reference_latent['samples'] must be BCHW; got shape {tuple(samples.shape)}")
            ref_samples_cpu = process_latent_for_model(model, samples).detach().to(device="cpu").clone()
        elif reference_latent is not None:
            raise RuntimeError("reference_latent must be a ComfyUI LATENT dict with a tensor 'samples' entry.")

        model_clone = model.clone()
        dm = safe_get_diffusion_model(model_clone)
        axes_dim = safe_axes_dim(dm)
        if not axes_dim:
            axes_dim = tuple()

        model_clone.model_options = clone_model_options(getattr(model_clone, "model_options", {}) or {})
        old_wrapper = model_clone.model_options.get("model_function_wrapper", None)
        append_transformer_patch(model_clone, "attn1_patch", flux_untwist_attn1_patch)
        method = normalize_reference_method(reference_latents_method)

        base_high_start = clamp_float(high_scale_start, -4.0, 8.0, 0.25)
        base_high_end = clamp_float(high_scale_end, -4.0, 8.0, 0.75)
        base_low_start = clamp_float(low_scale_start, -4.0, 8.0, 1.0)
        base_low_end = clamp_float(low_scale_end, -4.0, 8.0, 1.4)
        base_beta = clamp_float(beta, 0.01, 32.0, 2.0)
        base_adain = clamp_float(qk_adain_strength, 0.0, 1.0, 0.0)

        def model_function_wrapper(apply_model, args: Dict[str, Any]):
            input_x = args["input"]
            timestep = args["timestep"]
            c = dict(args["c"])
            cond_or_uncond = args.get("cond_or_uncond", None)

            progress = progress_from_timestep(timestep)
            active, _t = schedule_fraction(progress, start_percent, end_percent)

            to = dict(c.get("transformer_options", {}) or {})
            existing_refs = normalize_ref_latents(c.get("ref_latents", None))
            refs: List[torch.Tensor] = []

            if ref_samples_cpu is not None and active:
                ref = ref_samples_cpu.to(device=input_x.device, dtype=input_x.dtype)
                ref = repeat_to_batch(ref, int(input_x.shape[0]))
                refs.append(ref)

            if existing_refs:
                for ref in existing_refs:
                    refs.append(repeat_to_batch(ref.to(device=input_x.device, dtype=input_x.dtype), int(input_x.shape[0])))

            cfg = UntwistConfig(
                enabled=bool(active and refs),
                axes_dim=axes_dim,
                high_scale_start=base_high_start,
                high_scale_end=base_high_end,
                low_scale_start=base_low_start,
                low_scale_end=base_low_end,
                beta=base_beta,
                start_percent=start_percent,
                end_percent=end_percent,
                start_single_block=start_single_block,
                end_single_block=end_single_block,
                qk_adain_strength=base_adain,
                reference_latents_method=method,
                progress=progress,
                verbose=node_verbose,
            )
            to["flux_untwist_rope"] = cfg.as_transformer_options()
            c["transformer_options"] = to

            if refs:
                c["ref_latents"] = refs
                if ref_samples_cpu is not None or "ref_latents_method" not in c:
                    c["ref_latents_method"] = method

            if node_verbose:
                ref_shapes = [tuple(r.shape) for r in refs]
                print(
                    f"{_PREFIX} call: progress={progress:.3f} active={active} refs={len(refs)} "
                    f"method={c.get('ref_latents_method', '<native>')} shapes={ref_shapes}"
                )

            if old_wrapper is not None:
                return old_wrapper(
                    apply_model,
                    {
                        "input": input_x,
                        "timestep": timestep,
                        "c": c,
                        "cond_or_uncond": cond_or_uncond,
                    },
                )
            return apply_model(input_x, timestep, **c)

        model_clone.set_model_unet_function_wrapper(model_function_wrapper)

        if node_verbose:
            print(f"{_PREFIX} patched Flux-like model: {type(dm).__name__}")
            print(f"{_PREFIX} axes_dim={axes_dim or '<unknown>'} blocks={start_single_block}..{end_single_block}")
            print(f"{_PREFIX} high={base_high_start:.3f}->{base_high_end:.3f} low={base_low_start:.3f}->{base_low_end:.3f} beta={base_beta:.3f}")

        return (model_clone,)


class MiniMaxH3UntwistRoPE:
    """Patch native MiniMax H3 visual-reference attention after split-half RoPE."""

    CATEGORY = "model_patches/Untwisting RoPE"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    DESCRIPTION = (
        "Applies frequency-aware RoPE scaling to selected native MiniMax H3 visual-reference keys. "
        "Ordinary image and pure-video refs are targeted by default; video+audio and H3 Continuum context require explicit opt-in."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "high_scale_start": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": -4.0,
                        "max": 8.0,
                        "step": 0.01,
                        "tooltip": "Highest-frequency H3 reference-key scale at the start of the active window.",
                    },
                ),
                "high_scale_end": (
                    "FLOAT",
                    {
                        "default": 1.00,
                        "min": -4.0,
                        "max": 8.0,
                        "step": 0.01,
                        "tooltip": "Highest-frequency H3 reference-key scale at the end of the active window.",
                    },
                ),
                "low_scale_start": (
                    "FLOAT",
                    {
                        "default": 1.00,
                        "min": -4.0,
                        "max": 8.0,
                        "step": 0.01,
                        "tooltip": "Lowest-frequency H3 reference-key scale at the start of the active window.",
                    },
                ),
                "low_scale_end": (
                    "FLOAT",
                    {
                        "default": 1.05,
                        "min": -4.0,
                        "max": 8.0,
                        "step": 0.01,
                        "tooltip": "Lowest-frequency H3 reference-key scale at the end of the active window.",
                    },
                ),
                "beta": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.01,
                        "max": 32.0,
                        "step": 0.01,
                        "tooltip": "Polynomial interpolation exponent across each scaled H3 RoPE frequency bank. Paper default is 2.",
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Start of the active denoising-progress window. 0 is the first sampling step.",
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 0.90,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "End of the active denoising-progress window. Progress is derived from the actual sampler schedule so the final 10% is genuinely native.",
                    },
                ),
                "verbose": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "reference_scope": (
                    ["image_only", "image_and_video", "all_visual_including_continuum"],
                    {
                        "default": "image_and_video",
                        "tooltip": (
                            "Which native visual references may be untwisted. image_and_video includes ordinary image and pure-video refs only. "
                            "image_only restricts modulation to image refs. all_visual_including_continuum also includes video+audio and Continuum context and is experimental."
                        ),
                    },
                ),
                "scale_temporal_axis": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Experimental. Also scale H3's temporal RoPE bank. Off keeps the H3 t bank native and applies the frequency schedule only to H/W; "
                            "this is an H3 runtime default rather than a paper-derived temporal policy."
                        ),
                    },
                ),
            },
        }

    def patch(
        self,
        model: Any,
        high_scale_start: float,
        high_scale_end: float,
        low_scale_start: float,
        low_scale_end: float,
        beta: float,
        start_percent: float,
        end_percent: float,
        verbose: bool = False,
        reference_scope: str = "image_and_video",
        scale_temporal_axis: bool = False,
    ):
        node_verbose = coerce_bool(verbose)
        start_percent, end_percent = normalize_percent_window(start_percent, end_percent)
        scope = str(reference_scope or "image_and_video")
        if scope not in {"image_only", "image_and_video", "all_visual_including_continuum"}:
            scope = "image_and_video"
        temporal_axis = coerce_bool(scale_temporal_axis)

        model_clone = model.clone()
        dm = safe_get_minimax_h3_model(model_clone)

        blocks = getattr(dm, "blocks", None)
        if blocks is None or len(blocks) == 0:
            raise RuntimeError("MiniMax H3 Untwist RoPE requires a native H3 DiT with at least one transformer block.")
        first_attn = getattr(blocks[0], "attn", None)
        head_dim = int(getattr(first_attn, "head_dim", 0) or 0)
        if head_dim <= 0:
            raise RuntimeError("Could not determine MiniMax H3 attention head dimension.")

        rope = getattr(dm, "rope", None)
        inv_freq = getattr(rope, "inv_freq", None)
        if not torch.is_tensor(inv_freq) or inv_freq.numel() <= 0:
            raise RuntimeError("Could not determine MiniMax H3 RoPE frequency layout from model.rope.inv_freq.")

        rope_axis_count = 3
        rope_freqs_per_axis = int(inv_freq.numel())
        rotated_dim = 2 * rope_axis_count * rope_freqs_per_axis
        if rotated_dim > head_dim:
            raise RuntimeError(
                "MiniMax H3 RoPE layout is incompatible with this patch: "
                f"rotated_dim={rotated_dim} exceeds head_dim={head_dim}."
            )

        base_high_start = clamp_float(high_scale_start, -4.0, 8.0, 0.95)
        base_high_end = clamp_float(high_scale_end, -4.0, 8.0, 1.0)
        base_low_start = clamp_float(low_scale_start, -4.0, 8.0, 1.0)
        base_low_end = clamp_float(low_scale_end, -4.0, 8.0, 1.05)
        base_beta = clamp_float(beta, 0.01, 32.0, 2.0)

        model_clone.model_options = clone_model_options(getattr(model_clone, "model_options", {}) or {})
        old_wrapper = model_clone.model_options.get("model_function_wrapper", None)

        globally_neutral = (
            base_high_start == 1.0
            and base_high_end == 1.0
            and base_low_start == 1.0
            and base_low_end == 1.0
        )
        if globally_neutral:
            if node_verbose:
                print(f"{_H3_PREFIX} all scale endpoints are 1.0; returning an exact model no-op")
            return (model_clone,)

        model_clone.model_options, spectrum_instance_id = register_spectrum_h3_profile(
            model_clone.model_options,
            block_count=len(blocks),
            high_scale_start=base_high_start,
            high_scale_end=base_high_end,
            low_scale_start=base_low_start,
            low_scale_end=base_low_end,
            beta=base_beta,
            start_percent=start_percent,
            end_percent=end_percent,
            scope=scope,
            scale_temporal_axis=temporal_axis,
        )
        base_transformer_options = model_clone.model_options.setdefault("transformer_options", {})
        patch_time_attention_override = base_transformer_options.get("optimized_attention_override", None)

        def model_function_wrapper(apply_model, args: Dict[str, Any]):
            input_x = args["input"]
            timestep = args["timestep"]
            c = dict(args["c"])

            incoming_to = dict(c.get("transformer_options", {}) or {})
            # Current ComfyUI publishes the complete sampler schedule here. The
            # per-call `sigmas` transformer option is only the current coordinate
            # and therefore must not be mistaken for the schedule.
            sample_sigmas = incoming_to.get("sample_sigmas", None)
            if sample_sigmas is None:
                sample_sigmas = args.get("sigmas", None)
            progress = progress_from_schedule_index(timestep, sigmas=sample_sigmas)
            active, _t = schedule_fraction(progress, start_percent, end_percent)

            payload = c.get("minimax_payload", None)
            selection = minimax_h3_visual_reference_selection(payload, scope=scope)
            ref_ranges = list(selection.ranges)

            cfg = MiniMaxH3UntwistConfig(
                enabled=bool(active and selection.mapping_valid and ref_ranges),
                reference_ranges=tuple(ref_ranges),
                reference_scope=scope,
                rope_axis_count=rope_axis_count,
                rope_freqs_per_axis=rope_freqs_per_axis,
                high_scale_start=base_high_start,
                high_scale_end=base_high_end,
                low_scale_start=base_low_start,
                low_scale_end=base_low_end,
                beta=base_beta,
                start_percent=start_percent,
                end_percent=end_percent,
                scale_temporal_axis=temporal_axis,
                progress=progress,
                verbose=node_verbose,
            )

            to = append_spectrum_h3_runtime(
                incoming_to,
                instance_id=spectrum_instance_id,
                progress=progress,
                active=cfg.enabled,
            )
            if cfg.enabled:
                previous_attention_override = to.get(
                    "optimized_attention_override",
                    patch_time_attention_override,
                )
                to["optimized_attention_override"] = make_minimax_h3_attention_override(
                    previous_attention_override
                )
            to["minimax_h3_untwist_rope"] = cfg.as_transformer_options()
            c["transformer_options"] = to

            if node_verbose:
                ref_tokens = sum(end - start for start, end in ref_ranges)
                mapping = "ok" if selection.mapping_valid else f"invalid:{selection.reason}"
                print(
                    f"{_H3_PREFIX} call: progress={progress:.3f} active={active} enabled={cfg.enabled} "
                    f"scope={scope} temporal_axis={temporal_axis} mapping={mapping} "
                    f"native_visual_refs={selection.total_visual_refs} selected={len(ref_ranges)} "
                    f"selected_kinds={list(selection.selected_kinds)} skipped_scope={selection.skipped_video_refs} "
                    f"skipped_continuum={selection.skipped_continuum_refs} ref_tokens={ref_tokens} ranges={ref_ranges}"
                )

            next_args = dict(args)
            next_args["c"] = c
            if old_wrapper is not None:
                return old_wrapper(apply_model, next_args)
            return apply_model(input_x, timestep, **c)

        model_clone.set_model_unet_function_wrapper(model_function_wrapper)

        if node_verbose:
            print(f"{_H3_PREFIX} patched native H3 model: {type(dm).__name__}")
            print(
                f"{_H3_PREFIX} blocks={len(blocks)} head_dim={head_dim} "
                f"rotary_axes={rope_axis_count} freqs_per_axis={rope_freqs_per_axis} "
                f"rotated_dim={rotated_dim} unrotated_tail={head_dim - rotated_dim}"
            )
            print(
                f"{_H3_PREFIX} high={base_high_start:.3f}->{base_high_end:.3f} "
                f"low={base_low_start:.3f}->{base_low_end:.3f} beta={base_beta:.3f} "
                f"window={start_percent:.2f}->{end_percent:.2f} scope={scope} temporal_axis={temporal_axis} "
                f"spectrum_profile={spectrum_instance_id}"
            )

        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "Flux2UntwistRoPE": Flux2UntwistRoPE,
    "MiniMaxH3UntwistRoPE": MiniMaxH3UntwistRoPE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2UntwistRoPE": "Flux.2 Untwist RoPE",
    "MiniMaxH3UntwistRoPE": "MiniMax H3 Untwist RoPE",
}
