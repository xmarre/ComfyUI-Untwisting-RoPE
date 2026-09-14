from pathlib import Path


UTILS = Path("flux_untwist/utils.py")
NODES = Path("nodes.py")


utils = UTILS.read_text(encoding="utf-8")
needle = "\n\ndef _find_diffusion_model(model_patcher: Any, predicate: Any, error_message: str) -> Any:\n"
if needle not in utils:
    raise SystemExit("utils insertion anchor not found")
helper = r'''

_H3_FLOW_SAMPLING_CONTEXT_KEY = "h3_flow_sampling_context"


def progress_from_h3_flow_sampling_context(timestep: Any, context: Any) -> float | None:
    """Resolve H3 Untwist progress from Flow's diagnostic full-trajectory contract.

    The contract is opt-in and does not replace ComfyUI's sampler-owned
    ``sample_sigmas``.  When absent, callers must retain the legacy local
    schedule behavior.  A recognized but malformed contract fails closed.
    """
    if context is None:
        return None
    if not isinstance(context, dict):
        raise RuntimeError("h3_flow_sampling_context must be a dictionary")
    if context.get("api") != 1:
        raise RuntimeError("unsupported h3_flow_sampling_context API")
    if context.get("source") != "h3_flow_execution_contract_untwist_clock_trial":
        raise RuntimeError("unsupported h3_flow_sampling_context source")
    if context.get("units") != "comfy_sigma":
        raise RuntimeError("h3_flow_sampling_context must use comfy_sigma units")

    stage = str(context.get("stage", ""))
    if stage not in {"low", "probe", "high"}:
        raise RuntimeError("h3_flow_sampling_context has an invalid Flow stage")
    digest = context.get("original_schedule_digest")
    if not isinstance(digest, str) or len(digest) != 16:
        raise RuntimeError("h3_flow_sampling_context has an invalid original schedule digest")
    generation = context.get("invocation_generation")
    if not isinstance(generation, int) or generation <= 0:
        raise RuntimeError("h3_flow_sampling_context has an invalid invocation generation")

    raw_sigmas = context.get("original_nonzero_sigmas")
    try:
        if torch.is_tensor(raw_sigmas):
            schedule = raw_sigmas.detach().to(device="cpu", dtype=torch.float64).flatten()
        elif isinstance(raw_sigmas, (list, tuple)):
            schedule = torch.tensor([float(value) for value in raw_sigmas], dtype=torch.float64)
        else:
            raise TypeError
    except Exception as exc:
        raise RuntimeError("h3_flow_sampling_context has an invalid original sigma schedule") from exc
    if schedule.numel() < 2 or not bool(torch.isfinite(schedule).all().item()):
        raise RuntimeError("h3_flow_sampling_context requires at least two finite denoiser coordinates")
    if bool(torch.any(schedule <= 0).item()):
        raise RuntimeError("h3_flow_sampling_context original denoiser coordinates must be positive")
    if bool(torch.any(schedule[:-1] <= schedule[1:]).item()):
        raise RuntimeError("h3_flow_sampling_context original denoiser coordinates must be strictly descending")

    start_index = context.get("original_stage_start_index")
    if not isinstance(start_index, int) or not 0 <= start_index < int(schedule.numel()):
        raise RuntimeError("h3_flow_sampling_context has an invalid stage start index")
    if stage == "low" and start_index != 0:
        raise RuntimeError("h3_flow_sampling_context low stage must start at original index zero")

    current = _scalar_timestep_value(timestep)
    if not math.isfinite(current):
        raise RuntimeError("h3_flow_sampling_context current coordinate is non-finite")
    deltas = (schedule - current).abs()
    index = int(torch.argmin(deltas).item())
    delta = float(deltas[index].item())
    tolerance = max(1e-7, abs(current) * 1e-6)
    if delta > tolerance:
        raise RuntimeError(
            "h3_flow_sampling_context current coordinate is not on the original trajectory: "
            f"coordinate={current:.9g} nearest={float(schedule[index].item()):.9g} delta={delta:.3g}"
        )
    if index < start_index:
        raise RuntimeError("h3_flow_sampling_context current coordinate precedes the declared stage start")
    if stage == "probe" and index != start_index:
        raise RuntimeError("h3_flow_sampling_context probe must remain at its original handoff coordinate")

    return max(0.0, min(1.0, index / float(schedule.numel() - 1)))
'''
utils = utils.replace(needle, helper + needle, 1)
UTILS.write_text(utils, encoding="utf-8")

nodes = NODES.read_text(encoding="utf-8")
import_needle = "        progress_from_schedule_index,\n        progress_from_timestep,\n"
if nodes.count(import_needle) != 2:
    raise SystemExit(f"unexpected progress import count: {nodes.count(import_needle)}")
nodes = nodes.replace(
    import_needle,
    "        progress_from_h3_flow_sampling_context,\n        progress_from_schedule_index,\n        progress_from_timestep,\n",
)

progress_needle = '''            sample_sigmas = incoming_to.get("sample_sigmas", None)\n            if sample_sigmas is None:\n                sample_sigmas = args.get("sigmas", None)\n            progress = progress_from_schedule_index(timestep, sigmas=sample_sigmas)\n            active, _t = schedule_fraction(progress, start_percent, end_percent)\n'''
if nodes.count(progress_needle) != 1:
    raise SystemExit(f"unexpected H3 progress block count: {nodes.count(progress_needle)}")
progress_replacement = '''            sample_sigmas = incoming_to.get("sample_sigmas", None)\n            if sample_sigmas is None:\n                sample_sigmas = args.get("sigmas", None)\n            flow_sampling_context = incoming_to.get("h3_flow_sampling_context", None)\n            flow_progress = progress_from_h3_flow_sampling_context(timestep, flow_sampling_context)\n            if flow_progress is None:\n                progress = progress_from_schedule_index(timestep, sigmas=sample_sigmas)\n                progress_source = "local_sample_sigmas"\n            else:\n                progress = flow_progress\n                progress_source = "h3_flow_sampling_context_v1"\n            active, _t = schedule_fraction(progress, start_percent, end_percent)\n'''
nodes = nodes.replace(progress_needle, progress_replacement, 1)

cfg_needle = '''            to["minimax_h3_untwist_rope"] = cfg.as_transformer_options()\n            c["transformer_options"] = to\n'''
if nodes.count(cfg_needle) != 1:
    raise SystemExit(f"unexpected H3 runtime config block count: {nodes.count(cfg_needle)}")
cfg_replacement = '''            runtime_cfg = cfg.as_transformer_options()\n            runtime_cfg["progress_source"] = progress_source\n            if flow_sampling_context is not None:\n                runtime_cfg["flow_stage"] = str(flow_sampling_context["stage"])\n                runtime_cfg["flow_schedule_digest"] = str(flow_sampling_context["original_schedule_digest"])\n                runtime_cfg["flow_stage_start_index"] = int(flow_sampling_context["original_stage_start_index"])\n                runtime_cfg["flow_invocation_generation"] = int(flow_sampling_context["invocation_generation"])\n            to["minimax_h3_untwist_rope"] = runtime_cfg\n            c["transformer_options"] = to\n'''
nodes = nodes.replace(cfg_needle, cfg_replacement, 1)

verbose_needle = '''                    f"selected_kinds={list(selection.selected_kinds)} skipped_scope={selection.skipped_video_refs} "\n                    f"skipped_continuum={selection.skipped_continuum_refs} ref_tokens={ref_tokens} ranges={ref_ranges}"\n'''
if nodes.count(verbose_needle) != 1:
    raise SystemExit(f"unexpected H3 verbose block count: {nodes.count(verbose_needle)}")
verbose_replacement = '''                    f"selected_kinds={list(selection.selected_kinds)} skipped_scope={selection.skipped_video_refs} "\n                    f"skipped_continuum={selection.skipped_continuum_refs} ref_tokens={ref_tokens} ranges={ref_ranges} "\n                    f"progress_source={progress_source}"\n'''
nodes = nodes.replace(verbose_needle, verbose_replacement, 1)
NODES.write_text(nodes, encoding="utf-8")
