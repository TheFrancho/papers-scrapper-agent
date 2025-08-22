from pathlib import Path
from typing import Dict, Any, List

def _fmt_sched(sch):
    if isinstance(sch, dict):
        t = sch.get("type","cosine").lower()
        if t == "cosine": return "cosine annealing"
        steps = sch.get("steps", [])
        gamma = sch.get("drop_factor", 0.2)
        return f"step decay (milestones={steps}, gamma={gamma})"
    return str(sch)


def compose_wiki(spec: Dict[str, Any], code_paths: Dict[str, str], out_path: Path) -> None:
    lines = []
    lines.append("# Paper → Code Wiki")
    lines.append("")
    lines.append("## Dataset")
    ds = spec.get("dataset", {})
    lines += [
        f"- Name: {ds.get('name')}",
        f"- Num classes: {ds.get('num_classes')}",
        f"- Input size (C,H,W): {ds.get('input_size')}",
        f"- Optimizer: {ds.get('optimizer')}  lr={ds.get('lr')} momentum={ds.get('momentum')} weight_decay={ds.get('weight_decay')}",
        f"- Scheduler: {_fmt_sched(ds.get('scheduler'))}  epochs={ds.get('epochs')}  batch_size={ds.get('batch_size')}",
        ""
    ]

    lines.append("## Preprocessing")
    pp = spec.get("preprocess", {})
    norm = (pp.get("normalize") or {})
    aug  = (pp.get("augment") or {})
    lines += [
        f"- Normalize: mean={norm.get('mean')} std={norm.get('std')}",
        f"- Augment: crop={aug.get('random_crop')} padding={aug.get('padding')} flip={aug.get('random_flip')} cutout={aug.get('cutout')}",
        ""
    ]

    lines.append("## Model")
    md = spec.get("model", {})
    lines += [
        f"- Family: {md.get('family')} depth={md.get('depth')} widen_factor={md.get('widen_factor')} dropout={md.get('dropout')}",
        ""
    ]

    lines.append("## Training")
    tr = spec.get("train", {})
    lines += [
        f"- Optimizer: {tr.get('optimizer')} lr={tr.get('lr')} momentum={tr.get('momentum')} weight_decay={tr.get('weight_decay')}",
        f"- Scheduler: {tr.get('scheduler')} epochs={tr.get('epochs')} batch_size={tr.get('batch_size')}",
        ""
    ]

    lines.append("## Generated Code Artifacts")
    for k, p in sorted(code_paths.items()):
        lines.append(f"- `{k}` → `{p}`")
    lines.append("")

    cits: List[Dict[str, Any]] = spec.get("citations") or []
    if cits:
        lines.append("## Citations (paper excerpts supporting decisions)")
        for c in (cits or [])[:10]:
            lines.append(f"- **{c.get('section','(unknown)')}**: “{c.get('quote','').strip()}”")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
